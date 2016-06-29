-- Source: https://github.com/soumith/imagenet-multiGPU.torch/blob/master/util.lua

local ffi=require 'ffi'
------ Some FFI stuff used to pass storages between threads ------------------
ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THLongStorage_free(THLongStorage *self);
]]

local function setFloatStorage(tensor, storage_p)
   assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
   local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
   if cstorage ~= nil then
      ffi.C['THFloatStorage_free'](cstorage)
   end
   local storage = ffi.cast('THFloatStorage*', storage_p)
   tensor:cdata().storage = storage
end

local function setLongStorage(tensor, storage_p)
   assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
   local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
   if cstorage ~= nil then
      ffi.C['THLongStorage_free'](cstorage)
   end
   local storage = ffi.cast('THLongStorage*', storage_p)
   tensor:cdata().storage = storage
end

function sendTensor(inputs)
   local size = inputs:size()
   local ttype = inputs:type()
   local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
   inputs:cdata().storage = nil
   return {i_stg, size, ttype}
end

function receiveTensor(obj, buffer)
   local pointer = obj[1]
   local size = obj[2]
   local ttype = obj[3]
   if buffer then
      buffer:resize(size)
      assert(buffer:type() == ttype, 'Buffer is wrong type')
   else
      buffer = torch[ttype].new():resize(size)
   end
   if ttype == 'torch.FloatTensor' then
      setFloatStorage(buffer, pointer)
   elseif ttype == 'torch.LongTensor' then
      setLongStorage(buffer, pointer)
   else
      error('Unknown type')
   end
   return buffer
end

--Reduce the memory consumption by model by sharing the buffers
function optimizeNet( model, inputSize )
   local optnet_loaded, optnet = pcall(require,'optnet')
   if optnet_loaded then
      local opts   = {inplace=true, mode='training', removeGradParams=false}
      local input  = torch.rand(2,3,inputSize,inputSize)
      if opt.cuda then
          input = input:cuda()
      end
      optnet.optimizeMemory(model, input, opts)
   else
      print("'optnet' package not found, install it to reduce the memory consumption.")
      print("Repo: https://github.com/fmassa/optimize-net")
   end
end

function makeDataParallel(model, nGPU)
   -- Wrap the model with DataParallelTable, if using more than one GPU
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
	    require ("dpnn")
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   return model
end
