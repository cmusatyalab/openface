-- 2015-08-09: Originally from https://github.com/facebook/fbnn/blob/master/fbnn/Optim.lua.
-- 2015-08-09: [Brandon Amos] Initial optimizeTriplet implementation.
-- 2016-01-04: [Bartosz Ludwiczuk] Substantial improvements to optimizeTriplet at
--             https://github.com/melgor/Triplet-Learning

local pl = require('pl.import_into')()

local SoftPNOptim, _ = torch.class('SoftPNOptim')
-- deepcopy routine that assumes the presence of a 'clone' method in user
-- data should be used to deeply copy. This matches the behavior of Torch
-- tensors.
local function deepcopy(x)
    local typename = type(x)
    if typename == "userdata" then
        return x:clone()
    end
    if typename == "table" then
        local retval = {}
        for k, v in pairs(x) do
            retval[deepcopy(k)] = deepcopy(v)
        end
        return retval
    end
    return x
end

local function get_device_for_module(mod)
    local dev_id = nil
    for _, val in pairs(mod) do
        if torch.typename(val) == 'torch.CudaTensor' then
            local this_dev = val:getDevice()
            if this_dev ~= 0 then
                -- _make sure the tensors are allocated consistently
                assert(dev_id == nil or dev_id == this_dev)
                dev_id = this_dev
            end
        end
    end
    return dev_id -- _may still be zero if none are allocated.
end

local function on_device_for_module(mod, f)
    local this_dev = get_device_for_module(mod)
    if this_dev ~= nil then
        return cutorch.withDevice(this_dev, f)
    end
    return f()
end


-- Returns weight parameters and bias parameters and associated grad parameters
-- for this module. Annotates the return values with flag marking parameter set
-- as bias parameters set
function SoftPNOptim.weight_bias_parameters(module)
    local weight_params, bias_params
    if module.weight then
        weight_params = { module.weight, module.gradWeight }
        weight_params.is_bias = false
    end
    if module.bias then
        bias_params = { module.bias, module.gradBias }
        bias_params.is_bias = true
    end
    return { weight_params, bias_params }
end

function SoftPNOptim:__init(model, optState, checkpoint_data)
    assert(model)
    assert(checkpoint_data or optState)
    assert(not (checkpoint_data and optState))

    self.model = model
    self.modulesToOptState = {}
    -- Keep this around so we update it in setParameters
    self.originalOptState = optState

    -- Each module has some set of parameters and grad parameters. Since
    -- they may be allocated discontinuously, we need separate optState for
    -- each parameter tensor. self.modulesToOptState maps each module to
    -- a lua table of optState clones.
    if not checkpoint_data then
        self.model:apply(function(module)
            self.modulesToOptState[module] = {}
            local params = self.weight_bias_parameters(module)
            if pl.tablex.size(params) == 0 or pl.tablex.size(params) == 2 then
                for i, _ in ipairs(params) do
                    self.modulesToOptState[module][i] = deepcopy(optState)
                    if params[i] and params[i].is_bias then
                        -- never regularize biases
                        self.modulesToOptState[module][i].weightDecay = 0.0
                    end
                end
                assert(module)
                assert(self.modulesToOptState[module])
            end
        end)
    else
        local state = checkpoint_data.optim_state
        local modules = {}
        self.model:apply(function(m) table.insert(modules, m) end)
        assert(pl.tablex.compare_no_order(modules, pl.tablex.keys(state)))
        self.modulesToOptState = state
    end
    return self
end


local function createExtraModel(size)


    local mlp = nn.Sequential()

    -- get feature distances
    local cc = nn.ConcatTable()

    -- feats 1 with 3
    local cnn_left = nn.Sequential()
    local cnnpos_dist = nn.ConcatTable()
    cnnpos_dist:add(nn.SelectTable(1))
    cnnpos_dist:add(nn.SelectTable(3))
    cnn_left:add(cnnpos_dist)
    cnn_left:add(nn.PairwiseDistance(2))
    cnn_left:add(nn.View(size, 1))

    cc:add(cnn_left)

    -- feats 3 with 2
    local cnn_left2 = nn.Sequential()
    local cnnpos_dist2 = nn.ConcatTable()
    cnnpos_dist2:add(nn.SelectTable(3))
    cnnpos_dist2:add(nn.SelectTable(2))
    cnn_left2:add(cnnpos_dist2)
    cnn_left2:add(nn.PairwiseDistance(2))
    cnn_left2:add(nn.View(size, 1))

    cc:add(cnn_left2)

    -- feats 1 with 2
    local cnn_right = nn.Sequential()
    local cnnneg_dist = nn.ConcatTable()
    cnnneg_dist:add(nn.SelectTable(1))
    cnnneg_dist:add(nn.SelectTable(2))
    cnn_right:add(cnnneg_dist)
    cnn_right:add(nn.PairwiseDistance(2))
    cnn_right:add(nn.View(size, 1))

    cc:add(cnn_right)

    mlp:add(cc)

    local last_layer = nn.ConcatTable()

    -- select min negative distance inside the triplet
    local mined_neg = nn.Sequential()
    local mining_layer = nn.ConcatTable()
    mining_layer:add(nn.SelectTable(1))
    mining_layer:add(nn.SelectTable(2))
    mined_neg:add(mining_layer)
    mined_neg:add(nn.JoinTable(2))
    mined_neg:add(nn.Min(2))
    mined_neg:add(nn.View(size, 1))
    last_layer:add(mined_neg)
    -- add positive distance
    local pos_layer = nn.Sequential()
    pos_layer:add(nn.SelectTable(3))
    pos_layer:add(nn.View(size, 1))
    last_layer:add(pos_layer)

    mlp:add(last_layer)

    mlp:add(nn.JoinTable(2))
    return mlp
end

function SoftPNOptim:optimize(optimMethod, inputs, output, criterion, mapper) --, averageUse)
    assert(optimMethod)
    assert(inputs)
    assert(criterion)
    assert(self.modulesToOptState)
    local numImages = inputs:size(1)

    local extraModel = createExtraModel(output[1]:size(1))
    if opt.cuda then
        extraModel:cuda()
    end

    self.model:zeroGradParameters()

    local res = extraModel:forward(output)

    local err = criterion:forward(res, 1)
    local df_do = criterion:backward(res)
    df_do = extraModel:backward(output, df_do)

    --map gradient to the index of input
    local gradient_all = torch.Tensor(numImages, output[1]:size(2)):type(inputs:type())
    gradient_all:zero()
    --get all gradient for each example

    for i = 1, table.getn(mapper) do
        gradient_all[mapper[i][1]]:add(df_do[1][i])
        gradient_all[mapper[i][2]]:add(df_do[2][i])
        gradient_all[mapper[i][3]]:add(df_do[3][i])
    end

    --get average gradient per example: Not sure if it is right idea, so now Turn Off
    --   for i=1,numImages do
    --       if averageUse[i] ~= 0 then gradient_all[i]:div(averageUse[i])  end
    --   end
    --   print (('Gradient Average: %f: '):format(torch.abs(gradient_all):sum()))
    self.model:backward(inputs, gradient_all)

    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(_)
        return err, curGrad
    end

    for curMod, opt in pairs(self.modulesToOptState) do
        on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)
            if pl.tablex.size(curModParams) == 0 or
                    pl.tablex.size(curModParams) == 2 then
                if curModParams then
                    for i, _ in ipairs(curModParams) do
                        if curModParams[i] then
                            -- expect param, gradParam pair
                            curParam, curGrad = table.unpack(curModParams[i])
                            assert(curParam and curGrad)
                            optimMethod(fEvalMod, curParam, opt[i])
                        end
                    end
                end
            end
        end)
    end

    return err, output
end

return SoftPNOptim
