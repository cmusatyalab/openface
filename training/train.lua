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

local sanitize = paths.dofile('sanitize.lua')

local optimMethod = optim.adadelta
local optimState = {} -- Use for other algorithms like SGD
local optimator = OpenFaceOptim(model, optimState)

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local batchNumber
local triplet_loss


-- From https://groups.google.com/d/msg/torch7/i8sJYlgQPeA/wiHlPSa5-HYJ
local function replaceModules(net, orig_class_name, replacer)
   local nodes, container_nodes = net:findModules(orig_class_name)
   for i = 1, #nodes do
      for j = 1, #(container_nodes[i].modules) do
         if container_nodes[i].modules[j] == nodes[i] then
            local orig_mod = container_nodes[i].modules[j]
            container_nodes[i].modules[j] = replacer(orig_mod)
         end
      end
   end
end

local function cudnn_to_nn(net)
   local net_nn = net:clone():float()

   replaceModules(net_nn, 'cudnn.SpatialConvolution',
                  function(cudnn_mod)
                     local nn_mod = nn.SpatialConvolutionMM(
                        cudnn_mod.nInputPlane, cudnn_mod.nOutputPlane,
                        cudnn_mod.kW, cudnn_mod.kH,
                        cudnn_mod.dW, cudnn_mod.dH,
                        cudnn_mod.padW, cudnn_mod.padH
                     )
                     nn_mod.weight:copy(cudnn_mod.weight)
                     nn_mod.bias:copy(cudnn_mod.bias)
                     return nn_mod
                  end
   )
   replaceModules(net_nn, 'cudnn.SpatialAveragePooling',
                  function(cudnn_mod)
                     return nn.SpatialAveragePooling(
                        cudnn_mod.kW, cudnn_mod.kH,
                        cudnn_mod.dW, cudnn_mod.dH,
                        cudnn_mod.padW, cudnn_mod.padH
                     )
                  end
   )
   replaceModules(net_nn, 'cudnn.SpatialMaxPooling',
                  function(cudnn_mod)
                     return nn.SpatialMaxPooling(
                        cudnn_mod.kW, cudnn_mod.kH,
                        cudnn_mod.dW, cudnn_mod.dH,
                        cudnn_mod.padW, cudnn_mod.padH
                     )
                  end
   )

   replaceModules(net_nn, 'cudnn.ReLU', function() return nn.ReLU() end)
   replaceModules(net_nn, 'cudnn.SpatialCrossMapLRN',
                  function(cudnn_mod)
                     return nn.SpatialCrossMapLRN(cudnn_mod.size, cudnn_mod.alpha,
                                                  cudnn_mod.beta, cudnn_mod.K)
                  end
   )

   return net_nn
end

function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   if opt.cuda then
      cutorch.synchronize()
   end

   -- set the dropouts to training mode
   model:training()
   if opt.cuda then
      model:cuda() -- get it back on the right GPUs.
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

   local nnModel = sanitize(cudnn_to_nn(model)):float()
   torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), nnModel)
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
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
    for j = 1,n-1 do --for every image in batch
      local aIdx = embStartIdx + j - 1
      local diff = embeddings - embeddings[{ {aIdx} }]:expandAs(embeddings)
      local norms = diff:norm(2, 2):pow(2):squeeze()    --L2 norm have be squared
      for pair = j,n-1 do --create all posible positive pairs
        local pIdx = embStartIdx + pair
        -- Select a semi-hard negative that has a distance
        -- further away from the positive exemplar. Oxford-Face Idea

        --choose random example which is in margin
        local fff = (embeddings[aIdx]-embeddings[pIdx]):norm(2)
        local normsP = norms - torch.Tensor(embeddings:size(1)):fill(fff*fff)  --L2 norm have be squared
        --clean the idx of same class by setting to them max value
        normsP[{{embStartIdx,embStartIdx +n-1}}] = normsP:max()
        -- get indexes of example which are inside margin
        local in_margin = normsP:lt(opt.alpha)
        local allNeg = torch.find(in_margin, 1)

        if table.getn(allNeg) ~= 0 then  --use only non-random triplets. Random triples (which are beyond margin) will just produce gradient = 0, so average gradient will decrease
          selNegIdx = allNeg[math.random (table.getn(allNeg))]
          --get embeding of each example
          table.insert(as_table,embeddings[aIdx])
          table.insert(ps_table,embeddings[pIdx])
          table.insert(ns_table,embeddings[selNegIdx])
          -- get original idx of triplets
          table.insert(triplet_idx,{aIdx,pIdx,selNegIdx})
          -- increase number of times of using each example, need for averaging then
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
