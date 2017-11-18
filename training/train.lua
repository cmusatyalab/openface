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
local classificationOptim = require 'ClassificationOptim'
local siameseOptim = require 'SiameseOptim'
local distinceRatioOptim = require 'DistanceRatioOptim'
local hingeOptim = require 'HingeOptim'
local klDivOptim = require 'KLDivOptim'
local lmnnOptim = require 'LMNNOptim'
local softPNOptim = require 'SoftPNOptim'
local histogramOptim = require 'HistogramOptim'
local tEntropyOptim = require 'TEntropyOptim'
local optimMethod = optim.adam
local optimState = {} -- Use for other algorithms like SGD
local optimator

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

local batchNumber
local triplet_loss

function train()
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)
    batchNumber = 0
    -- 'crossentropy' 'kldiv'
    -- 's_cosine' 's_hinge' 's_double_margin' 's_global'
    -- 't_orj' 't_improved' 't_global' 'dist_ratio'
    -- 'lsss' 'lmnn' 'softPN' 'histogram' 'quadruplet'

    model, criterion = models.modelSetup(model)
    if opt.criterion == 'crossentropy' or opt.criterion == 'margin' or opt.criterion == 'lsss' or opt.criterion == 'multi' then
        optimator = classificationOptim:__init(model, optimState)
    elseif opt.criterion == 'kldiv' then
        optimator = klDivOptim:__init(model, optimState)
    elseif opt.criterion == 's_cosine' or opt.criterion == 's_global' or opt.criterion == 's_hadsell' or opt.criterion == 's_double_margin' then
        optimator = siameseOptim:__init(model, optimState)
    elseif opt.criterion == 's_hinge' then
        optimator = hingeOptim:__init(model, optimState)
    elseif opt.criterion == 't_orj' or opt.criterion == 't_improved' or opt.criterion == 't_global' or opt.criterion == 'lsss' then
        optimator = openFaceOptim:__init(model, optimState)
    elseif opt.criterion == 'lmnn' then
        optimator = lmnnOptim:__init(model, optimState)
    elseif opt.criterion == 'dist_ratio' then
        optimator = distinceRatioOptim:__init(model, optimState)
    elseif opt.criterion == 'softPN' then
        optimator = softPNOptim:__init(model, optimState)
    elseif opt.criterion == 'histogram' then
        optimator = histogramOptim:__init(model, optimState)
    elseif opt.criterion == 't_entropy' then
        optimator = tEntropyOptim:__init(model, optimState)
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
    print(opt.save)
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
    local saved_model = model:clone()
    cleanupModel(saved_model)
    torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), saved_model:clearState():float())
    torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

    if dpt then -- OOM without this
        dpt:clearState()
    end

    collectgarbage()
    return model
end

function zeroDataSize(data)
    if type(data) == 'table' then
        for i = 1, #data do
            data[i] = zeroDataSize(data[i])
        end
    elseif type(data) == 'userdata' then
        data = torch.Tensor():typeAs(data)
    end
    return data
end

-- Resize the output, gradInput, etc temporary tensors to zero (so that the
-- on disk size is smaller)
function cleanupModel(node)
    if node.output ~= nil then
        node.output = zeroDataSize(node.output)
    end
    if node.gradInput ~= nil then
        node.gradInput = zeroDataSize(node.gradInput)
    end
    if node.finput ~= nil then
        node.finput = zeroDataSize(node.finput)
    end
    -- Recurse on nodes with 'modules'
    if (node.modules ~= nil) then
        if (type(node.modules) == 'table') then
            for i = 1, #node.modules do
                local child = node.modules[i]
                cleanupModel(child)
            end
        end
    end

    collectgarbage()
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

    local inputs, error
    if opt.cuda then
        inputs = inputsCPU:cuda()
    else
        inputs = inputsCPU
    end
    local embeddings = model:forward(inputs):float()

    function optimize()
        local err, _
        -- 'crossentropy' 'kldiv'
        -- 's_cosine' 's_hinge' 's_double_margin' 's_global'
        -- 't_orj' 't_improved' 't_global' 'dist_ratio'
        -- 'lsss' 'lmnn' 'softPN' 'histogram' 'quadruplet'

        if opt.criterion == 'crossentropy' or opt.criterion == 'margin' or opt.criterion == 'lsss' or opt.criterion == 'multi' then
            err, _ = optimator:optimize(optimMethod, inputs, embeddings, targets, criterion)
        elseif opt.criterion == 'kldiv' or opt.criterion == 's_double_margin' or opt.criterion == 's_hadsell' then
            local as, targets, mapper = pairss(embeddings, numPerClass[1], 1, 0)
            err, _ = optimator:optimize(optimMethod, inputs, as, targets, criterion, mapper)
        elseif opt.criterion == 's_cosine' or opt.criterion == 's_hinge' or opt.criterion == 's_global' or opt.criterion == 'histogram' then
            local as, targets, mapper = pairss(embeddings, numPerClass[1], 1, -1)
            err, _ = optimator:optimize(optimMethod, inputs, as, targets, criterion, mapper)
        elseif opt.criterion == 't_orj' or opt.criterion == 't_improved' or opt.criterion == 't_global' or opt.criterion == 'dist_ratio' or opt.criterion == 'softPN' or opt.criterion == 'lsss' then

            local apn, triplet_idx = triplets(embeddings, inputs:size(1), numPerClass)
            if apn == nil then
                return
            else
                err, _ = optimator:optimize(optimMethod, inputs, apn, criterion, triplet_idx)
            end
        elseif opt.criterion == 't_entropy' then

            local apn, triplet_idx = triplets(embeddings, inputs:size(1), numPerClass)
            if apn == nil then
                return
            else
                err, _ = optimator:optimize(optimMethod, inputs, embeddings, apn, targets, criterion, triplet_idx)
            end
        elseif opt.criterion == 'lmnn' then
            local apn, triplet_idx = LMNNTriplets(embeddings, inputs:size(1), numPerClass)
            err, _ = optimator:optimize(optimMethod, inputs, apn, criterion, triplet_idx)
        end

        return err
    end

    error = optimize()
    if error == nil then
        return
    end

    if opt.cuda then
        cutorch.synchronize()
    end

    batchNumber = batchNumber + 1
    print(('Epoch: [%d][%d/%d]\tTime %.3f\tErr %.2e'):format(epoch, batchNumber, opt.epochSize, timer:time().real, error))
    timer:reset()
    triplet_loss = triplet_loss + error
end
