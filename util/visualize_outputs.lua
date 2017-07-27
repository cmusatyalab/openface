#!         /usr/bin/env th
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
photo = "angry.0013"
cmd:option('-imgPath', "/home/cenk/Documents/openface-v2/gamo/data/notaligned48/train/angry/angry.0013.png",
    'Path to aligned image.')
cmd:option('-filterOutput',
    'outputs/' .. photo,
    'Output directory.')
cmd:option('-model', '/media/cenk/DISK_5TB/losses/results/gamo/notaligned64_128/crossentropy/nn4/model_187.t7', 'Path to model.')
cmd:option('-imgDim', 64, 'Image dimension. nn1=224, nn4=96')
cmd:option('-numPreview', 8, 'Number of images to preview')
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
local img_orig = image.load(opt.imgPath, 3)
img[1] = image.scale(img_orig, opt.imgDim, opt.imgDim)
net:forward(img)

local fName = opt.filterOutput .. '/preview.html'
print("Outputting filter preview to '" .. fName .. "'")
f, err = io.open(fName, 'w')
if err then
    print("Error: Unable to open preview.html");
    os.exit(-1)
end

k = 0
torch.IntTensor({ 1, 5, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 }):apply(function(x)
    print(x)
    k = k + 1
    return x
end):apply(function(i)
    os.execute(string.format('mkdir -p %s/%s',
        opt.filterOutput, i))
    out = net.modules[i].output[1]
    f:write(string.format("<h6>Layer %s", i))
    --f:write(string.format("<h1>Layer %s</h1>\n", i))
    for j = 1, out:size(1) do
        imgName = string.format('%s/%d/%d.png',
            opt.filterOutput, i, j)
        image.save(imgName, out[j])
        if j <= opt.numPreview then
            f:write(string.format("<img src='%d/%d.png' width='64px'></img>",
                i, j))
        end
    end
    f:write("\n")
end)
