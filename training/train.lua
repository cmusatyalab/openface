-- Copyright 2015-2016 Carnegie Mellon University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- 2015-08-09: [Brandon Amos] Initial implementation.
-- 2016-01-04: [Bartosz Ludwiczuk] Substantial improvements at
--             https://github.com/melgor/Triplet-Learning

require 'optim'
require 'image'
require 'torchx' --for concetration the table of tensors

paths.dofile("OpenFaceOptim.lua")

local optimMethod = optim.adam
local optimState = {} -- Use for other algorithms like SGD
local optimator = OpenFaceOptim(model, optimState)

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local batchNumber
local triplet_loss

function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   batchNumber = 0
   if opt.cuda then
      cutorch.synchronize()
   end

   model:training()
   if opt.cuda then
      model:cuda()
      if opt.cudnn then
        cudnn.convert(model,cudnn)
      end
   end
   local tm = torch.Timer()
   triplet_loss = 0

   local i = 1
   while batchNumber < opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, numPerClass = trainLoader:samplePeople(opt.peoplePerBatch,
                                                                 opt.imagesPerPerson)
            inputs = inputs:float()
            numPerClass = numPerClass:float()
            return sendTensor(inputs), sendTensor(numPerClass)
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
      if i % 5 == 0 then
         donkeys:synchronize()
      end
      i = i + 1
   end

   donkeys:synchronize()
   if opt.cuda then
      cutorch.synchronize()
   end

   triplet_loss = triplet_loss / batchNumber

   trainLogger:add{
      ['avg triplet loss (train set)'] = triplet_loss,
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average triplet loss (per batch): %.2f',
                       epoch, tm:time().real, triplet_loss))
   print('\n')

   collectgarbage()

   -- Check for nans from https://github.com/cmusatyalab/openface/issues/127
   local function checkNans(x, tag)
      local I = torch.ne(x,x)
      if torch.any(I) then
         print("train.lua: Error: NaNs found in: ", tag)
         os.exit(-1)
         -- x[I] = 0.0
      end
   end

   for j, mod in ipairs(model:listModules()) do
      if torch.typename(mod) == 'nn.SpatialBatchNormalization' then
         checkNans(mod.running_mean, string.format("%d-%s-%s", j, mod, 'running_mean'))
         checkNans(mod.running_var, string.format("%d-%s-%s", j, mod, 'running_var'))
      end
   end

   if opt.cudnn then
      cudnn.convert(model, nn)
   end
   model = model:float():clearState()

   torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

   if opt.cuda then
      model = model:cuda()
      if opt.cudnn then
         cudnn.convert(model, cudnn)
      end
   end
   collectgarbage()
end -- of train()

local inputsCPU = torch.FloatTensor()
local numPerClass = torch.FloatTensor()

local timer = torch.Timer()
function trainBatch(inputsThread, numPerClassThread)
  if batchNumber >= opt.epochSize then
    return
  end

  if opt.cuda then
    cutorch.synchronize()
  end
  timer:reset()
  receiveTensor(inputsThread, inputsCPU)
  receiveTensor(numPerClassThread, numPerClass)

  local inputs
  if opt.cuda then
     inputs = inputsCPU:cuda()
  else
     inputs = inputsCPU
  end

  local numImages = inputs:size(1)
  local embeddings = model:forward(inputs):float()

  local as_table = {}
  local ps_table = {}
  local ns_table = {}

  local triplet_idx = {}
  local num_example_per_idx = torch.Tensor(embeddings:size(1))
  num_example_per_idx:zero()

  local tripIdx = 1
  local embStartIdx = 1
  local numTrips = 0
  for i = 1,opt.peoplePerBatch do
    local n = numPerClass[i]
    for j = 1,n-1 do -- For every image in the batch.
      local aIdx = embStartIdx + j - 1
      local diff = embeddings - embeddings[{ {aIdx} }]:expandAs(embeddings)
      local norms = diff:norm(2, 2):pow(2):squeeze()
      for pair = j, n-1 do -- For every possible positive pair.
        local pIdx = embStartIdx + pair

        local fff = (embeddings[aIdx]-embeddings[pIdx]):norm(2)
        local normsP = norms - torch.Tensor(embeddings:size(1)):fill(fff*fff)

        -- Set the indices of the same class to the max so they are ignored.
        normsP[{{embStartIdx,embStartIdx +n-1}}] = normsP:max()

        -- Get indices of images within the margin.
        local in_margin = normsP:lt(opt.alpha)
        local allNeg = torch.find(in_margin, 1)

        -- Use only non-random triplets.
        -- Random triples (which are beyond the margin) will just produce gradient = 0,
        -- so the average gradient will decrease.
        if table.getn(allNeg) ~= 0 then
          selNegIdx = allNeg[math.random (table.getn(allNeg))]
          -- Add the embeding of each example.
          table.insert(as_table,embeddings[aIdx])
          table.insert(ps_table,embeddings[pIdx])
          table.insert(ns_table,embeddings[selNegIdx])
          -- Add the original index of triplets.
          table.insert(triplet_idx, {aIdx,pIdx,selNegIdx})
          -- Increase the number of times of using each example.
          num_example_per_idx[aIdx] = num_example_per_idx[aIdx] + 1
          num_example_per_idx[pIdx] = num_example_per_idx[pIdx] + 1
          num_example_per_idx[selNegIdx] = num_example_per_idx[selNegIdx] + 1
          tripIdx = tripIdx + 1
        end

        numTrips = numTrips + 1
      end
    end
    embStartIdx = embStartIdx + n
  end
  assert(embStartIdx - 1 == numImages)
  local nTripsFound = table.getn(as_table)
  print(('  + (nTrips, nTripsFound) = (%d, %d)'):format(numTrips, nTripsFound))

  if nTripsFound == 0 then
     print("Warning: nTripsFound == 0. Skipping batch.")
     return
  end

  local as = torch.concat(as_table):view(table.getn(as_table), opt.embSize)
  local ps = torch.concat(ps_table):view(table.getn(ps_table), opt.embSize)
  local ns = torch.concat(ns_table):view(table.getn(ns_table), opt.embSize)

  local apn
  if opt.cuda then
     local asCuda = torch.CudaTensor()
     local psCuda = torch.CudaTensor()
     local nsCuda = torch.CudaTensor()

     local sz = as:size()
     asCuda:resize(sz):copy(as)
     psCuda:resize(sz):copy(ps)
     nsCuda:resize(sz):copy(ns)

     apn = {asCuda, psCuda, nsCuda}
  else
     apn = {as, ps, ns}
  end

  local err, _ = optimator:optimizeTriplet(
     optimMethod, inputs, apn, criterion,
     triplet_idx -- , num_example_per_idx
  )

  -- DataParallelTable's syncParameters
  model:apply(function(m) if m.syncParameters then m:syncParameters() end end)
  if opt.cuda then
     cutorch.synchronize()
  end
  batchNumber = batchNumber + 1
  print(('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %.2e'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err))
  timer:reset()
  triplet_loss = triplet_loss + err
end
