--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 04.04.2017
-- Time: 18:50
-- To change this template use File | Settings | File Templates.
--

require 'nn'
require 'torch'
require 'dpnn'
require 'optim'
require 'image'
require 'torchx'
require 'optim'
require 'xlua'


torch.setdefaulttensortype("torch.FloatTensor")
opt = { embSize = 128 }

function visualize()
    net = createModel()
    input = torch.rand(10, 3, imgDim, imgDim)

    generateGraph = require 'optnet.graphgen'

    graphOpts = {
        displayProps = { shape = 'ellipse', fontsize = 12, style = 'solid' },
        nodeData = function(oldData, tensor)
            return oldData .. '\n' .. 'Size: ' .. tensor:numel()
        end
    }

    g = generateGraph(net, input, graphOpts)

    graph.dot(g, 'models/' .. modelname .. '_optimized_' .. imgDim, 'models/' .. modelname .. '_optimized_' .. imgDim)
end

data = { "alexnet", "nn4", "vgg-face" }
imgDims = { 28, 32, 64 }
for j = 1, #imgDims do
    imgDim = imgDims[j]
    for i = 1, #data do

        modelname = data[i]
        print(imgDim, modelname)
        paths.dofile('/home/cenk/Documents/openface-v2/models/mine/' .. imgDim .. '/' .. modelname .. '.def.lua')
        visualize()
    end
end
