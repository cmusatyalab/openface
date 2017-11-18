local ffi = require 'ffi'

local batchNumber, nImgs = 0

torch.setdefaulttensortype('torch.FloatTensor')

function batchRepresent()
   local loadSize   = {opt.channelSize, opt.imgDim, opt.imgDim}
   print(opt.data)
   local cacheFile = paths.concat(opt.data, 'cache.t7')
   print('cache lotation: ', cacheFile)
   local dumpLoader
   if paths.filep(cacheFile) then
      print('Loading metadata from cache.')
      print('If your dataset has changed, delete the cache file.')
      dumpLoader = torch.load(cacheFile)
   else
      print('Creating metadata for cache.')
      dumpLoader = dataLoader{
         paths = {opt.data},
         loadSize = loadSize,
         sampleSize = loadSize,
         split = 0,
         verbose = true
      }
      torch.save(cacheFile, dumpLoader)
   end
   collectgarbage()
   nImgs = dumpLoader:sizeTest()
   print('nImgs: ', nImgs)
   assert(nImgs > 0, "Failed to get nImgs")

   batchNumber = 0

   for i=1,math.ceil(nImgs/opt.batchSize) do
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = math.min(nImgs, indexStart + opt.batchSize - 1)
      local batchSz = indexEnd-indexStart+1
      local inputs, labels = dumpLoader:get(indexStart, indexEnd)
      local paths = {}
      for j=indexStart,indexEnd do
         table.insert(paths,
                      ffi.string(dumpLoader.imagePath[dumpLoader.testIndices[j]]:data()))
      end
      repBatch(paths, inputs, labels, batchSz)
      if i % 5 == 0 then
         collectgarbage()
      end
   end

   if opt.cuda then
      cutorch.synchronize()
   end
end

function repBatch(paths, inputs, labels, batchSz)
   batchNumber = batchNumber + batchSz

   if opt.cuda then
      inputs = inputs:cuda()
   end
   local embeddings = model:forward(inputs):float()
   if opt.cuda then
      cutorch.synchronize()
   end

   if batchSz == 1 then
      embeddings = embeddings:reshape(1, embeddings:size(1))
   end

   for i=1,batchSz do
      labelsCSV:write({labels[i], paths[i]})
      repsCSV:write(embeddings[i]:totable())
   end

   print(('Represent: %d/%d'):format(batchNumber, nImgs))
end
