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
local testPy = opt.testPy

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
    local outDir = paths.concat(opt.save, 'rep-' .. epoch)
    print(latestModelFile)
    print(outDir)
    local batch_cmd = batchRepresent
    if opt.cuda then
        assert(opt.device ~= nil)
        batch_cmd = batch_cmd .. ' -cuda -device ' .. opt.device .. ' '
    end
    cmd = batch_cmd .. ' -batchSize ' .. opt.testBatchSize ..
            ' -model ' .. latestModelFile ..
            ' -data ' .. opt.data ..
            ' -outDir ' .. outDir .. '/train'..
            ' -imgDim ' .. opt.imgDim ..
            ' -channelSize ' .. opt.channelSize
    print(cmd)
    os.execute(cmd)

    cmd = batch_cmd .. ' -batchSize ' .. opt.testBatchSize ..
            ' -model ' .. latestModelFile ..
            ' -data ' .. opt.testDir ..
            ' -outDir ' .. outDir .. '/test'..
            ' -imgDim ' .. opt.imgDim ..
            ' -channelSize ' .. opt.channelSize
    print(cmd)
    os.execute(cmd)

    cmd = 'python ' .. testPy .. ' --trainDir ' .. outDir .. '/train --testDir ' .. outDir .. '/test'
    print(cmd)
    os.execute(cmd)
    -- this is for pairs
    --    lfwAcc = getLfwAcc(paths.concat(outDir, "accuracies.txt"))
    --    testLogger:add {
    --        ['lfwAcc'] = lfwAcc
    --    }
end
