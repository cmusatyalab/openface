-- Model: nn4.def.lua
-- Description: Implementation of NN4 from the FaceNet paper.
-- Input size: 3x96x96
-- Number of Parameters from net:getParameters() with embSize=128: 6959088
-- Components: Mostly `nn`
-- Devices: CPU and CUDA
--
-- Brandon Amos <http://bamos.github.io>
-- 2015-09-18
--
-- Copyright 2015-2016 Carnegie Mellon University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

imgDim = 96

function createModel()
   local net = nn.Sequential()

   net:add(nn.SpatialConvolutionMM(3, 64, 7, 7, 2, 2, 3, 3))
   net:add(nn.SpatialBatchNormalization(64))
   net:add(nn.ReLU())

   -- The FaceNet paper just says `norm` and that the models are based
   -- heavily on the inception paper (http://arxiv.org/pdf/1409.4842.pdf),
   -- which uses pooling and normalization in the same way in the early layers.
   --
   -- The Caffe and official versions of this network both use LRN:
   --
   --   + https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
   --   + https://github.com/google/inception/blob/master/inception.ipynb
   --
   -- The Caffe docs at http://caffe.berkeleyvision.org/tutorial/layers.html
   -- define LRN to be across channels.
   net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
   net:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))

   -- Inception (2)
   net:add(nn.SpatialConvolutionMM(64, 64, 1, 1))
   net:add(nn.SpatialBatchNormalization(64))
   net:add(nn.ReLU())
   net:add(nn.SpatialConvolutionMM(64, 192, 3, 3, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(192))
   net:add(nn.ReLU())

   net:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
   net:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

   -- Inception (3a)
   net:add(nn.Inception{
     inputSize = 192,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 32},
     reduceSize = {96, 16, 32, 64},
     pool = nn.SpatialMaxPooling(3, 3, 2, 2),
     batchNorm = true
   })

   -- Inception (3b)
   net:add(nn.Inception{
     inputSize = 256,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {128, 64},
     reduceSize = {96, 32, 64, 64},
     pool = nn.SpatialLPPooling(256, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (3c)
   net:add(nn.Inception{
     inputSize = 320,
     kernelSize = {3, 5},
     kernelStride = {2, 2},
     outputSize = {256, 64},
     reduceSize = {128, 32, nil, nil},
     pool = nn.SpatialMaxPooling(3, 3, 2, 2),
     batchNorm = true
   })

   -- Inception (4a)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {192, 64},
     reduceSize = {96, 32, 128, 256},
     pool = nn.SpatialLPPooling(640, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (4b)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {224, 64},
     reduceSize = {112, 32, 128, 224},
     pool = nn.SpatialLPPooling(640, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (4c)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {256, 64},
     reduceSize = {128, 32, 128, 192},
     pool = nn.SpatialLPPooling(640, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (4d)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {1, 1},
     outputSize = {288, 64},
     reduceSize = {144, 32, 128, 160},
     pool = nn.SpatialLPPooling(640, 2, 3, 3),
     batchNorm = true
   })

   -- Inception (4e)
   net:add(nn.Inception{
     inputSize = 640,
     kernelSize = {3, 5},
     kernelStride = {2, 2},
     outputSize = {256, 128},
     reduceSize = {160, 64, nil, nil},
     pool = nn.SpatialMaxPooling(3, 3, 2, 2),
     batchNorm = true
   })

   -- Inception (5a)
   net:add(nn.Inception{
              inputSize = 1024,
              kernelSize = {3},
              kernelStride = {1},
              outputSize = {384},
              reduceSize = {192, 128, 384},
              pool = nn.SpatialLPPooling(960, 2, 3, 3),
              batchNorm = true
   })

   -- Inception (5b)
   net:add(nn.Inception{
              inputSize = 896,
              kernelSize = {3},
              kernelStride = {1},
              outputSize = {384},
              reduceSize = {192, 128, 384},
              pool = nn.SpatialMaxPooling(3, 3, 2, 2),
              batchNorm = true
   })

   net:add(nn.SpatialAveragePooling(3, 3))

   -- Validate shape with:
   -- net:add(nn.Reshape(896))

   net:add(nn.View(896))
   net:add(nn.Linear(896, opt.embSize))
   net:add(nn.Normalize(2))

   return net
end
