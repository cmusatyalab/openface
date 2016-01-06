require 'nn'

require 'cunn'
require 'dpnn'

require 'fbnn'
require 'fbcunn'

require 'optim'

require 'cudnn'

cudnn.benchmark = false
cudnn.fastest = true
cudnn.verbose = false

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
   local net_nn = net:clone():float()

   replaceModules(net_nn, 'nn.SpatialConvolutionMM',
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
   replaceModules(net_nn, 'nn.SpatialAveragePooling',
                  function(nn_mod)
                     return cudnn.SpatialAveragePooling(
                        nn_mod.kW, nn_mod.kH,
                        nn_mod.dW, nn_mod.dH,
                        nn_mod.padW, nn_mod.padH
                     )
                  end
   )
   replaceModules(net_nn, 'nn.SpatialMaxPooling',
                  function(nn_mod)
                     return cudnn.SpatialMaxPooling(
                        nn_mod.kW, nn_mod.kH,
                        nn_mod.dW, nn_mod.dH,
                        nn_mod.padW, nn_mod.padH
                     )
                  end
   )

   replaceModules(net_nn, 'nn.ReLU', function() return cudnn.ReLU() end)
   replaceModules(net_nn, 'nn.SpatialCrossMapLRN',
                  function(nn_mod)
                     return cudnn.SpatialCrossMapLRN(cudnn_mod.size, cudnn_mod.alpha,
                                                     cudnn_mod.beta, cudnn_mod.k)
                  end
   )

   return net_nn
end

if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = torch.load(opt.retrain)
else
   paths.dofile(opt.modelDef)
   model = createModel()
end

model = nn_to_cudnn(model)
criterion = nn.TripletEmbeddingCriterion(opt.alpha)

model = model:cuda()
criterion:cuda()

print('=> Model')
print(model)

print('=> Criterion')
print(criterion)

collectgarbage()
