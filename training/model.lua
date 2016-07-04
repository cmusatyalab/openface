require 'nn'

require 'dpnn'

require 'optim'

if opt.cuda then
   require 'cunn'
   if opt.cudnn then
      require 'cudnn'
      cudnn.benchmark = opt.cudnn_bench
      cudnn.fastest = true
      cudnn.verbose = false
   end
end

paths.dofile('torch-TripletEmbedding/TripletEmbedding.lua')


local M = {}

function M.modelSetup(continue)
  if continue then
     model = continue
  elseif opt.retrain ~= 'none' then
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
  
  -- First remove any DataParallelTable
  if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
  end
  
  criterion = nn.TripletEmbeddingCriterion(opt.alpha)
  
  if opt.cuda then
    model = model:cuda()
    if opt.cudnn then
      cudnn.convert(model,cudnn)
    end
    criterion:cuda()
  else
    model:float()
    criterion:float()
  end
  
  optimizeNet(model, opt.imgDim)
  
  if opt.cuda and opt.nGPU > 1 then
    model = makeDataParallel(model, opt.nGPU)
  end
  
  collectgarbage()
  return model, criterion
end

return M


