-- Copyright 2015 Carnegie Mellon University
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

testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local testDataIterator = function()
   testLoader:reset()
   return function() return testLoader:get_batch(false) end
end

local batchNumber
local triplet_loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   model:evaluate()
   model:cuda()

   triplet_loss = 0
   for i=1,opt.testEpochSize do
      donkeys:addjob(
         function()
            local inputs, labels = testLoader:sampleTriplet(opt.batchSize)
            inputs = inputs:float()
            return sendTensor(inputs)
         end,
         testBatch
      )
      if i % 5 == 0 then
         donkeys:synchronize()
         collectgarbage()
      end
   end

   donkeys:synchronize()
   cutorch.synchronize()

   triplet_loss = triplet_loss / opt.testEpochSize
   testLogger:add{
      ['avg triplet loss (test set)'] = triplet_loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average triplet loss (per batch): %.2f',
                       epoch, timer:time().real, triplet_loss))
   print('\n')


end

local inputsCPU = torch.FloatTensor()
local inputs = torch.CudaTensor()

function testBatch(inputsThread)
   receiveTensor(inputsThread, inputsCPU)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)

   local embeddings = model:forward({
         inputs:sub(1,opt.batchSize),
         inputs:sub(opt.batchSize+1, 2*opt.batchSize),
         inputs:sub(2*opt.batchSize+1, 3*opt.batchSize)})
   local err = criterion:forward(embeddings)
   cutorch.synchronize()

   triplet_loss = triplet_loss + err
   print(('Epoch: Testing [%d][%d/%d] Triplet Loss: %.2f'):format(epoch, batchNumber,
                                                                  opt.testEpochSize, err))
   batchNumber = batchNumber + 1
end
