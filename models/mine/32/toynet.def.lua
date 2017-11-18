--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 14.01.2017
-- Time: 09:49
-- To change this template use File | Settings | File Templates.
--

imgDim = 32


function createModel()

    local SpatialConvolution = nn.SpatialConvolutionMM
    local SpatialMaxPooling = nn.SpatialMaxPooling

    local net = nn.Sequential()
    net:add(SpatialConvolution(3, 32, 7, 7, 4, 4, 2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(32))
    net:add(SpatialMaxPooling(3, 3, 2, 2))
    net:add(SpatialConvolution(32, 96, 5, 5, 1, 1, 2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(96))
    net:add(SpatialMaxPooling(3, 3, 2, 2))
    net:add(SpatialConvolution(96, 192, 3, 3, 1, 1, 1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialBatchNormalization(192))
    net:add(nn.Dropout(0.5))
    net:add(nn.View(192 * 1))
    net:add(nn.Linear(192, opt.embSize))
    net:add(nn.Normalize(2))


    return net
end


