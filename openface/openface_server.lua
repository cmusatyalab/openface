#!/usr/bin/env th
--
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


require 'torch'
require 'nn'
require 'dpnn'
require 'image'

io.stdout:setvbuf 'no'
torch.setdefaulttensortype('torch.FloatTensor')

-- OpenMP-acceleration causes slower performance. Related issues:
-- https://groups.google.com/forum/#!topic/cmu-openface/vqkkDlbfWZw
-- https://github.com/torch/torch7/issues/691
-- https://github.com/torch/image/issues/7
torch.setnumthreads(1)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Face recognition server.')
cmd:text()
cmd:text('Options:')

cmd:option('-model', './models/openface/nn4.v1.t7', 'Path to model.')
cmd:option('-imgDim', 96, 'Image dimension. nn1=224, nn4=96')
cmd:option('-cuda', false)
cmd:text()

opt = cmd:parse(arg or {})
-- print(opt)

net = torch.load(opt.model)
net:evaluate()
-- print(net)

local imgCuda = nil
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   net = net:cuda()
   imgCuda = torch.CudaTensor(1, 3, opt.imgDim, opt.imgDim)
end

local img = torch.Tensor(1, 3, opt.imgDim, opt.imgDim)
while true do
   -- Read a path to an image on stdin and output the representation
   -- as a CSV.
   local imgPath = io.read("*line")
   if imgPath and imgPath:len() ~= 0 then
      img[1] = image.load(imgPath, 3, byte)
      img[1] = image.scale(img[1], opt.imgDim, opt.imgDim)
      local rep
      if opt.cuda then
         imgCuda:copy(img)
         rep = net:forward(imgCuda):float()
      else
         rep = net:forward(img)
      end
      local sz = rep:size(1)
      for i = 1,sz do
         io.write(rep[i])
         if i < sz then
            io.write(',')
         end
      end
      io.write('\n')
      io.stdout:flush()
   end
end
