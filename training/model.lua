require 'nn'

require 'dpnn'

require 'optim'

require 'cunn'

if opt.cudnn then
  require 'cudnn'
  cudnn.benchmark = opt.cudnn_bench
  cudnn.fastest = true
  cudnn.verbose = false
end
paths.dofile('torch-TripletEmbedding/TripletEmbedding.lua')


local M = {}

function M.modelSetup(continue)
  if opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain);
    model = torch.load(opt.retrain)
    print("Using imgDim = ", opt.imgDim)
  elseif continue then
    model = continue
  else
    paths.dofile(opt.modelDef)
    assert(imgDim, "Model definition must set global variable 'imgDim'")
    assert(imgDim == opt.imgDim, "Model definiton's imgDim must match imgDim option.")
    model = createModel()
    MSRinit(model)
    FCinit(model)
  end
  
  -- First remove any DataParallelTable
  if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
  end
  model = model:cuda()
  
  optimizeNet(model, opt.imgDim)
  
  if opt.cudnn then
    cudnn.convert(model,cudnn)
  end

  model = makeDataParallel(model, opt.nGPU)
  
  criterion = nn.TripletEmbeddingCriterion(opt.alpha)
  criterion:cuda()
  collectgarbage()
  return model, criterion
end

return M


