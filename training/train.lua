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
local optnet_loaded, optnet = pcall(require, 'optnet')
local models = require 'model'
local openFaceOptim = require 'OpenFaceOptim'
local softmaxOptim = require 'SoftmaxOptim'
local pairLossOptim = require 'PairLossOptim'

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
    model, criterion = models.modelSetup(model)
    if opt.criterion == 'loglikelihood' or opt.criterion == 'kl' then
        optimator = softmaxOptim:__init(model, optimState)
    elseif opt.criterion == 'cosine' or opt.criterion == 'l1hinge' or opt.criterion == 'l2loss' or opt.criterion == 'hinge' then
        optimator = pairLossOptim:__init(model, optimState)
    elseif opt.criterion == 'triplet' then
        optimator = openFaceOptim:__init(model, optimState)
    end

    if opt.cuda then
        cutorch.synchronize()
    end
    model:training()

    local tm = torch.Timer()
    triplet_loss = 0

    local i = 1
    while batchNumber < opt.epochSize do
        -- queue jobs to data-workers
        donkeys:addjob(-- the job callback (runs in data-worker thread)
            function()
                local inputs, numPerClass, targets = trainLoader:samplePeople(opt.peoplePerBatch, opt.imagesPerPerson)
                inputs = inputs:float()
                numPerClass = numPerClass:float()
                return sendTensor(inputs), sendTensor(numPerClass), sendTensor(targets)
            end,
            -- the end callback (runs in the main thread)
            trainBatch)
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

    trainLogger:add {
        ['avg triplet loss (train set)'] = triplet_loss,
    }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
            .. 'average triplet loss (per batch): %.2f',
        epoch, tm:time().real, triplet_loss))
    print('\n')

    collectgarbage()
end

-- of train()


function saveModel(model)
    -- Check for nans from https://github.com/cmusatyalab/openface/issues/127
    local function checkNans(x, tag)
        local I = torch.ne(x, x)
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
        dpt = model
        model = model:get(1)
    end


    if optnet_loaded then
        optnet.removeOptimization(model)
    end

    torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:clearState():float())
    torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

    if dpt then -- OOM without this
        dpt:clearState()
    end

    collectgarbage()

    return model
end

local inputsCPU = torch.FloatTensor()
local numPerClass = torch.FloatTensor()
local targets = torch.FloatTensor()

local timer = torch.Timer()
function trainBatch(inputsThread, numPerClassThread, targetsThread)
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
    receiveTensor(targetsThread, targets)

    local inputs
    if opt.cuda then
        inputs = inputsCPU:cuda()
    else
        inputs = inputsCPU
    end
    local embeddings
    if opt.criterion == 'hinge' then
        local as, targets, mapper = pairss(inputs, numPerClass[1])
        local n = 70
        for i = 0, (as[1]:size(1) / n) - 1 do

            local as1 = subrange(as[1], i * n + 1, (i + 1) * n)
            local as2 = subrange(as[2], i * n + 1, (i + 1) * n)
            local sub_targets = subrange(targets, i * n + 1, (i + 1) * n)
            embeddings = model:forward({ as1, as2 }):float()
            total_err, _ = optimator:optimize(optimMethod, { as1, as2 }, embeddings, sub_targets, criterion, mapper)
            if total_err == nil then
                return
            end
        end

    else
        embeddings = model:forward(inputs):float()
    end


    function optimize()
        local err, _ = nil, nil
        if opt.criterion == 'loglikelihood' or opt.criterion == 'kl' then
            err, _ = optimator:optimize(optimMethod, inputs, embeddings, targets, criterion)
        elseif opt.criterion == 'cosine' or opt.criterion == 'l1hinge' or opt.criterion == 'l2loss' then

            err, _ = optimator:optimize(optimMethod, as, embeddings, targets, criterion, mapper)
        elseif opt.criterion == 'triplet' then
            local apn, triplet_idx = triplets(embeddings, inputs:size(1), numPerClass)
            if apn == nil then
                return
            else
                err, _ = optimator:optimize(optimMethod, inputs, apn, criterion, triplet_idx)
            end
        end
        return err
    end

    if opt.criterion == 'hinge' then
        print('hinge')
    else
        total_err = optimize()
        if total_err == nil then
            return
        end
    end
    if opt.cuda then
        cutorch.synchronize()
    end

    batchNumber = batchNumber + 1
    print(('Epoch: [%d][%d/%d]\tTime %.3f\ttripErr %.2e'):format(epoch, batchNumber, opt.epochSize, timer:time().real, total_err))
    timer:reset()
    triplet_loss = triplet_loss + total_err
end
