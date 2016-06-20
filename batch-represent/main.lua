#!/usr/bin/env th

require 'torch'
require 'optim'

require 'paths'

require 'xlua'
require 'csvigo'

require 'nn'
require 'dpnn'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.device)
end

opt.manualSeed = 2
torch.manualSeed(opt.manualSeed)

paths.dofile('dataset.lua')
paths.dofile('batch-represent.lua')

model = torch.load(opt.model)
model:evaluate()
if opt.cuda then
   model:cuda()
end

repsCSV = csvigo.File(paths.concat(opt.outDir, "reps.csv"), 'w')
labelsCSV = csvigo.File(paths.concat(opt.outDir, "labels.csv"), 'w')

batchRepresent()

repsCSV:close()
labelsCSV:close()
