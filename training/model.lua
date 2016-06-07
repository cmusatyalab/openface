require 'nn'

require 'dpnn'

require 'optim'

if opt.cuda then
   require 'cunn'
   if opt.cudnn then
      require 'cudnn'
      cudnn.benchmark = false
      cudnn.fastest = true
      cudnn.verbose = false
   end
end

paths.dofile('torch-TripletEmbedding/TripletEmbedding.lua')

if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = torch.load(opt.retrain)
   print("Using imgDim = ", opt.imgDim)
else
   paths.dofile(opt.modelDef)
   assert(imgDim, "Model definition must set global variable 'imgDim'")
   assert(imgDim == opt.imgDim, "Model definiton's imgDim must match imgDim option.")
   model = createModel()
end

criterion = nn.TripletEmbeddingCriterion(opt.alpha)

if opt.cuda then
   model = model:cuda()
   if opt.cudnn then
    cudnn.convert(model,cudnn)
   end
   criterion:cuda()
end

-- optimizeNet(model, opt.imgDim)

print('=> Model')
print(model)
print(('Number of Parameters: %d'):format(model:getParameters():size(1)))

print('=> Criterion')
print(criterion)

collectgarbage()
