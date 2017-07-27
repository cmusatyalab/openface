#!     /usr/bin/env th

require 'torch'
require 'optim'

require 'paths'

require 'xlua'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

if opt.cuda then
    require 'cutorch'
    cutorch.setDevice(opt.device)
end

torch.save(paths.concat(opt.save, 'opts.t7'), opt, 'ascii')
print('Saving everything to: ' .. opt.save)

torch.setdefaulttensortype('torch.FloatTensor')

torch.manualSeed(opt.manualSeed)

paths.dofile('data.lua')
paths.dofile('util.lua')
model = nil
criterion = nil
paths.dofile('train.lua')
paths.dofile('test.lua')

paths.dofile('tripletSelection.lua')
paths.dofile('pairSelection.lua')
paths.dofile('lmnnTripletSelection.lua')
paths.dofile('criterion.lua')


if opt.peoplePerBatch > nClasses then
    print('\n\nError: opt.peoplePerBatch > number of classes. Please decrease this value.')
    print('  + opt.peoplePerBatch: ', opt.peoplePerBatch)
    print('  + number of classes: ', nClasses)
    os.exit(-1)
end

epoch = opt.epochNumber

for epo = 1, opt.nEpochs do
    train()
    if epo % 25 == 0 then
        model = saveModel(model)
        if opt.testing then
            print(epo)
            test()
        end
    end
    epoch = epoch + 1
end
