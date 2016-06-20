-- Copyright 2016 Carnegie Mellon University
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

require 'io'
require 'string'
require 'sys'

local batchRepresent = "../batch-represent/main.lua"
local lfwEval = "../evaluation/lfw.py"

local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local function getLfwAcc(fName)
    local f = io.open(fName, 'r')
    io.input(f)
    local lastLine = nil
    while true do
    local line = io.read("*line")
    if line == nil then break end
    lastLine = line
    end
    io.close()
    return tonumber(string.sub(lastLine, 6, 11))
end

function test()
   if opt.cuda then
      model = model:float()
   end
   local latestModelFile = paths.concat(opt.save, 'model_' .. epoch .. '.t7')
   local outDir = paths.concat(opt.save, 'lfw-' .. epoch)
   print(latestModelFile)
   print(outDir)
   local cmd = batchRepresent
   if opt.cuda then
      assert(opt.device ~= nil)
      cmd = cmd .. ' -cuda -device ' .. opt.device
   end
   cmd = cmd .. ' -batchSize ' .. opt.testBatchSize ..
      ' -model ' .. latestModelFile ..
      ' -data ' .. opt.lfwDir ..
      ' -outDir ' .. outDir ..
      ' -imgDim ' .. opt.imgDim
   os.execute(cmd)

   cmd = lfwEval .. ' Epoch' .. epoch .. ' ' .. outDir
   os.execute(cmd)

   lfwAcc = getLfwAcc(paths.concat(outDir, "accuracies.txt"))
   testLogger:add{
      ['lfwAcc'] = lfwAcc
   }
end
