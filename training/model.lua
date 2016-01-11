require 'nn'

require 'dpnn'
require 'fbnn'

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

-- From https://groups.google.com/d/msg/torch7/i8sJYlgQPeA/wiHlPSa5-HYJ
function replaceModules(net, orig_class_name, replacer)
   local nodes, container_nodes = net:findModules(orig_class_name)
   for i = 1, #nodes do
      for j = 1, #(container_nodes[i].modules) do
         if container_nodes[i].modules[j] == nodes[i] then
            local orig_mod = container_nodes[i].modules[j]
            container_nodes[i].modules[j] = replacer(orig_mod)
         end
      end
   end
end

function nn_to_cudnn(net)
   local net_cudnn = net:clone():float()

   replaceModules(net_cudnn, 'nn.SpatialConvolutionMM',
                  function(nn_mod)
                     local cudnn_mod = cudnn.SpatialConvolution(
                        nn_mod.nInputPlane, nn_mod.nOutputPlane,
                        nn_mod.kW, nn_mod.kH,
                        nn_mod.dW, nn_mod.dH,
                        nn_mod.padW, nn_mod.padH
                     )
                     cudnn_mod.weight:copy(nn_mod.weight)
                     cudnn_mod.bias:copy(nn_mod.bias)
                     return cudnn_mod
                  end
   )
   replaceModules(net_cudnn, 'nn.SpatialAveragePooling',
                  function(nn_mod)
                     return cudnn.SpatialAveragePooling(
                        nn_mod.kW, nn_mod.kH,
                        nn_mod.dW, nn_mod.dH,
                        nn_mod.padW, nn_mod.padH
                     )
                  end
   )
   replaceModules(net_cudnn, 'nn.SpatialMaxPooling',
                  function(nn_mod)
                     return cudnn.SpatialMaxPooling(
                        nn_mod.kW, nn_mod.kH,
                        nn_mod.dW, nn_mod.dH,
                        nn_mod.padW, nn_mod.padH
                     )
                  end
   )

   replaceModules(net_cudnn, 'nn.ReLU', function() return cudnn.ReLU() end)
   replaceModules(net_cudnn, 'nn.SpatialCrossMapLRN',
                  function(nn_mod)
                     return cudnn.SpatialCrossMapLRN(nn_mod.size, nn_mod.alpha,
                                                     nn_mod.beta, nn_mod.k)
                  end
   )

   return net_cudnn
end

if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = torch.load(opt.retrain)
else
   paths.dofile(opt.modelDef)
   model = createModel()
end

if opt.cudnn then
   model = nn_to_cudnn(model)
end
criterion = nn.TripletEmbeddingCriterion(opt.alpha)

if opt.cuda then
   model = model:cuda()
   criterion:cuda()
end

print('=> Model')
print(model)
print(('Number of Parameters: %d'):format(model:getParameters():size(1)))

print('=> Criterion')
print(criterion)

collectgarbage()
