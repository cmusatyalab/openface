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
local optnet_loaded, optnet = pcall(require,'optnet')
local models = require 'model'
local openFaceOptim = require 'OpenFaceOptim'


local optimMethod = optim.adam
local optimState = {} -- Use for other algorithms like SGD
local optimator = nil

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local batchNumber
local triplet_loss

function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   batchNumber = 0
   model,criterion = models.modelSetup(model)
   optimator = openFaceOptim:__init(model, optimState)
   if opt.cuda then
     cutorch.synchronize()
   end
   model:training()

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
end -- of train()


function saveModel(model)
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
   if opt.cuda then
    if opt.cudnn then
	cudnn.convert(model, nn)
    end
   end

   local dpt
   if torch.type(model) == 'nn.DataParallelTable' then
      dpt   = model
      model = model:get(1)
   end


   if optnet_loaded then
    optnet.removeOptimization(model)
   end

   torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'),  model:float():clearState())
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

   if dpt then -- OOM without this
      dpt:clearState()
   end

   collectgarbage()

   return model
end

local inputsCPU = torch.FloatTensor()
local numPerClass = torch.FloatTensor()

local timer = torch.Timer()
function trainBatch(inputsThread, numPerClassThread)
  collectgarbage()
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
  apn,triplet_idx =  triplets(embeddings,numImages,numPerClass)

  local err, _ = optimator:optimizeTriplet(
     optimMethod, inputs, apn, criterion, triplet_idx -- , num_example_per_idx
  )
  if opt.cuda then
    cutorch.synchronize()
  end

  batchNumber = batchNumber + 1
  print(('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %.2e'):format(
        epoch, batchNumber, opt.epochSize, timer:time().real, err))
  timer:reset()
  triplet_loss = triplet_loss + err
end
