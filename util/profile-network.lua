#!/usr/bin/env th
--
-- Outputs the number of parameters in a network for a single image
-- in evaluation mode.

require 'torch'
require 'nn'
require 'dpnn'

torch.setdefaulttensortype('torch.FloatTensor')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Network Size.')
cmd:text()
cmd:text('Options:')

cmd:option('-model', '/media/cenk/DISK_5TB/losses/results/cife/notaligned64_128/crossentropy/alexnet/model_189.t7', 'Path to model.')
cmd:option('-imgDim', 64, 'Image dimension. nn1=224, nn4=96')
cmd:option('-numIter', 10)
cmd:option('-cuda', true)
cmd:text()

local opt = cmd:parse(arg or {})
-- print(opt)

local net = torch.load(opt.model):float()
net:evaluate()

local img = torch.randn(opt.numIter, 1, 3, opt.imgDim, opt.imgDim)

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   net = net:cuda()
   img = img:cuda()
end

local times = torch.Tensor(opt.numIter)

for i=1,opt.numIter do
   local timer = torch.Timer()
   local _ = net:forward(img[i])
   times[i] = 1000.0*timer:time().real
end

print(string.format('Single image forward pass: %.2f ms +/- %.2f ms',
                    torch.mean(times), torch.std(times)))
