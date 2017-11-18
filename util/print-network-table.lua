#!   /usr/bin/env th

require 'torch'
require 'nn'
require 'dpnn'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Print network table.')
cmd:text()
cmd:text('Options:')

cmd:option('-modelDef', '/home/cenk/Documents/openface-v2/models/mine/64/alexnet.def.lua', 'Path to model definition.')
cmd:option('-imgDim', 64, 'Image dimension. nn1=224, nn4=96')
cmd:option('-embSize', 128)
cmd:text()

opt = cmd:parse(arg or {})

paths.dofile(opt.modelDef)
local net = createModel()

local img = torch.randn(3, 3, opt.imgDim, opt.imgDim)
net:forward(img)
print(opt.modelDef)
for i=1,#net.modules do
   local module = net.modules[i]
   local out = torch.typename(module) .. ": "
   for _, sz in ipairs(torch.totable(module.output:size())) do
      out = out .. sz .. ', '
   end
   out = string.sub(out, 1, -3)
   print(out)
end
