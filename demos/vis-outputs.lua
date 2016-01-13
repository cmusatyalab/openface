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
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Visualize OpenFace outputs.')
cmd:text()
cmd:text('Options:')

cmd:option('-imgPath', 'images/examples-aligned/examples/lennon-1.png',
           'Path to aligned image.')
cmd:option('-filterOutput',
           'images/examples-aligned/examples/lennon-1',
           'Output directory.')
cmd:option('-model', './models/openface/nn4.small2.v1.t7', 'Path to model.')
cmd:option('-imgDim', 96, 'Image dimension. nn1=224, nn4=96')
cmd:option('-numPreview', 39, 'Number of images to preview')
cmd:text()

opt = cmd:parse(arg or {})
-- print(opt)

os.execute('mkdir -p ' .. opt.filterOutput)

if not paths.filep(opt.imgPath) then
   print("Unable to find: " .. opt.imgPath)
   os.exit(-1)
end

net = torch.load(opt.model)
net:evaluate()
print(net)

local img = torch.Tensor(1, 3, opt.imgDim, opt.imgDim)
img[1] = image.load(opt.imgPath, opt.imgDim)
img[1] = image.scale(img[1], opt.imgDim, opt.imgDim)
net:forward(img)

f, err = io.open(opt.filterOutput .. '/preview.html', 'w')
if err then
   print("Error: Unable to open preview.html");
   os.exit(-1)
end

torch.IntTensor({3, 7, 10, 11, 12, 13, 14, 15, 16,
                 17, 18, 19, 20, 21}):apply(function (i)
      os.execute(string.format('mkdir -p %s/%s',
                               opt.filterOutput, i))
      out = net.modules[i].output[1]
      f:write(string.format("<h1>Layer %s</h1>\n", i))
      for j = 1,out:size(1) do
         imgName = string.format('%s/%d/%d.png',
                                  opt.filterOutput, i, j)
         image.save(imgName, out[j])
         if j <= opt.numPreview then
            f:write(string.format("<img src='%d/%d.png' width='96px'></img>\n",
                                  i, j))
         end
      end
end)
