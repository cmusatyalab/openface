local lfs = require 'lfs'

local M = { }

-- http://stackoverflow.com/questions/6380820/get-containing-path-of-lua-file
function script_path()
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

function M.parse(arg)

   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('OpenFace')
   cmd:text()
   cmd:text('Options:')

   ------------ General options --------------------
   cmd:option('-cache',
              paths.concat(script_path(), 'work'),
              'subdirectory in which to save/log experiments')
   cmd:option('-data',
              paths.concat(os.getenv('HOME'), 'openface', 'data',
                           'casia-facescrub',
                           'dlib-affine-sz:96'),
                           -- 'dlib-affine-224-split'),
              'Home of dataset. Split into "train" and "val" directories that separate images by class.')
   cmd:option('-manualSeed', 2, 'Manually set RNG seed')
   cmd:option('-cuda', true, 'Use cuda.')
   cmd:option('-cudnn', true, 'Convert the model to cudnn.')

   ------------- Data options ------------------------
   cmd:option('-nDonkeys', 2, 'number of donkeys to initialize (data loading threads)')

   ------------- Training options --------------------
   cmd:option('-nEpochs', 1000, 'Number of total epochs to run')
   cmd:option('-epochSize', 500, 'Number of batches per epoch')
   cmd:option('-epochNumber', 1, 'Manual epoch number (useful on restarts)')
   -- GPU memory usage depends on peoplePerBatch and imagesPerPerson.
   cmd:option('-peoplePerBatch', 15, 'Number of people to sample in each mini-batch.')
   cmd:option('-imagesPerPerson', 20, 'Number of images to sample per person in each mini-batch.')
   cmd:option('-testBatchSize', 800, 'Batch size for testing.')

   ---------- Model options ----------------------------------
   cmd:option('-retrain', 'none', 'provide path to model to retrain with')
   cmd:option('-modelDef', '../models/openface/nn4.def.lua', 'path to model definiton')
   cmd:option('-imgDim', 96, 'Image dimension. nn2=224, nn4=96')
   cmd:option('-embSize', 128, 'size of embedding from model')
   cmd:option('-alpha', 0.2, 'margin in TripletLoss')
   cmd:text()

   local opt = cmd:parse(arg or {})
   os.execute('mkdir -p ' .. opt.cache)
   local count = 1
   for f in lfs.dir(opt.cache) do
      local isDir = paths.dirp(paths.concat(opt.cache, f))
      if f ~= "." and f ~= ".." and isDir then
         count = count + 1
      end
   end
   opt.save = paths.concat(opt.cache, string.format("%03d", count))

   return opt
end

return M
