#!/usr/bin/env th

require 'torch'
require 'cutorch'
require 'optim'

require 'paths'

require 'xlua'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

os.execute('mkdir -p ' .. opt.save)
torch.save(paths.concat(opt.save, 'opts.t7'), opt, 'ascii')
print('Saving everything to: ' .. opt.save)

torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(1)
torch.manualSeed(opt.manualSeed)

paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')
paths.dofile('util.lua')

if opt.peoplePerBatch > nClasses then
  print('\n\nError: opt.peoplePerBatch > number of classes. Please decrease this value.')
  print('  + opt.peoplePerBatch: ', opt.peoplePerBatch)
  print('  + number of classes: ', nClasses)
  os.exit(-1)
end

epoch = opt.epochNumber

-- test()
for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
end
