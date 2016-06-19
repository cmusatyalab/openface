-- Model: resnet1.def.lua
-- Description: ResNet model for face recognition with OpenFace, v1.
-- Input size: 3x96x96
-- Number of Parameters from net:getParameters() with embSize=128: TODO
-- Components: Mostly `nn`
-- Devices: CPU and CUDA
--
-- Brandon Amos <http://bamos.github.io>
-- 2016-06-19
--
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
--
-- Modified from:
-- https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
-- Portions from here are BSD licensed.

imgDim = 96

local nn = require 'nn'

local Convolution = nn.SpatialConvolutionMM
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

function createModel()
   local depth = 18
   local shortcutType = 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
                    :add(nn.Identity())
                    :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
                 :add(s)
                 :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
                 :add(s)
                 :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   -- Configurations for ResNet:
   --  num. residual blocks, num features, residual block function
   local cfg = {
      [18]  = {{2, 2, 2, 2}, 4608, basicblock},
      -- [34]  = {{3, 4, 6, 3}, 512, basicblock},
      -- [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
      -- [101] = {{3, 4, 23, 3}, 2048, bottleneck},
      -- [152] = {{3, 8, 36, 3}, 2048, bottleneck},
   }

   assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
   local def, nLinear, block = table.unpack(cfg[depth])
   iChannels = 64

   -- The ResNet ImageNet model
   local model = nn.Sequential()
   model:add(Convolution(3,64,7,7,2,2,3,3))
   model:add(SBatchNorm(64))
   model:add(ReLU(true))
   model:add(Max(3,3,2,2,1,1))
   model:add(layer(block, 64, def[1]))
   model:add(layer(block, 128, def[2], 2))
   model:add(layer(block, 256, def[3], 2))
   model:add(layer(block, 512, def[4], 2))
   -- TODO: Add back?
   -- model:add(Avg(7, 7, 1, 1))
   model:add(nn.View(nLinear))
   model:add(nn.Linear(nLinear, opt.embSize))
   model:add(nn.Normalize(2))

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('nn.SpatialConvolutionMM')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   return model
end
