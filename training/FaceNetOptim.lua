-- Modified from https://github.com/facebook/fbnn/blob/master/fbnn/Optim.lua.

local pl = require('pl.import_into')()

local FaceNetOptim, parent = torch.class('FaceNetOptim', 'nn.Optim')

function FaceNetOptim:__init(model, optState, checkpoint_data)
   parent.__init(self, model, optState, checkpoint_data)
end

local function get_device_for_module(mod)
   local dev_id = nil
   for name, val in pairs(mod) do
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

function FaceNetOptim:optimizeTriplet(optimMethod, inputs, criterion)
   assert(optimMethod)
   assert(inputs)
   assert(criterion)
   assert(self.modulesToOptState)

   self.model:zeroGradParameters()
   local output = self.model:forward(inputs)

   local err = criterion:forward(output)

   local df_do = criterion:backward(output)
   self.model:backward(inputs, df_do)

    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(x)
        return err, curGrad
    end

    for curMod, opt in pairs(self.modulesToOptState) do
       on_device_for_module(curMod, function()
                               local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
                               assert(pl.tablex.size(curModParams) == 0 or
                                         pl.tablex.size(curModParams) == 2)
            if curModParams then
               for i, tensor in ipairs(curModParams) do
                  if curModParams[i] then
                        -- expect param, gradParam pair
                     curParam, curGrad = table.unpack(curModParams[i])
                     assert(curParam and curGrad)
                     optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
       end)
    end

    return err, output
end
