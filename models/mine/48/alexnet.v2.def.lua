--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 14.01.2017
-- Time: 09:49
-- To change this template use File | Settings | File Templates.
--

imgDim = 48


function createModel()

    local SpatialConvolution = nn.SpatialConvolutionMM
    local SpatialMaxPooling = nn.SpatialMaxPooling

    local net = nn.Sequential()
    net:add(SpatialConvolution(1, 64, 7, 7, 2, 2, 3, 3)) -- Changed
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(SpatialMaxPooling(3, 3, 2, 2)) -- 55 ->  27
    net:add(SpatialConvolution(64, 192, 5, 5, 1, 1, 2, 2)) --  27 -> 27
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(192))
    net:add(SpatialMaxPooling(3, 3, 2, 2)) --  27 ->  13
    net:add(SpatialConvolution(192, 384, 3, 3, 1, 1, 1, 1)) --  13 ->  13
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(384))
    net:add(SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)) --  13 ->  13
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(256))
    net:add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)) --  13 ->  13
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(256))
    net:add(SpatialMaxPooling(2, 2, 1, 1)) -- Changed
    net:add(nn.View(256 * 4 * 4)) --Changed
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(256 * 4 * 4, 4096)) --Changed
    net:add(nn.ReLU(true))
    net:add(nn.BatchNormalization(4096))
    net:add(nn.Threshold(0, 1e-6))
    net:add(nn.Dropout(0.5))
    net:add(nn.Linear(4096, 4096))
    net:add(nn.ReLU(true))
    net:add(nn.BatchNormalization(4096))
    net:add(nn.Threshold(0, 1e-6))

    net:add(nn.Linear(4096, opt.embSize))
    net:add(nn.Normalize(2))


    return net
end


