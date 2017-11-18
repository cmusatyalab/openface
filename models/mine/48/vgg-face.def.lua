--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 14.01.2017
-- Time: 09:49
-- To change this template use File | Settings | File Templates.
--

imgDim = 48


function createModel()

    local conv = nn.SpatialConvolutionMM
    local relu = nn.ReLU
    local mp = nn.SpatialMaxPooling
    local net = nn.Sequential()

    net:add(conv(1, 64, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(conv(64, 64, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(mp(2, 2, 2, 2))
    net:add(conv(64, 128, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(conv(128, 128, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(128))
    net:add(mp(2, 2, 2, 2))
    net:add(conv(128, 256, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(256))
    net:add(conv(256, 256, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(256))
    net:add(conv(256, 256, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(256))
    net:add(mp(2, 2, 2, 2))
    net:add(conv(256, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(mp(2, 2, 2, 2))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(nn.SpatialBatchNormalization(512))
    net:add(mp(2, 2, 2, 2))

    net:add(nn.View(512 * 1 * 1)) --Changed
    net:add(nn.Linear(512 * 1 * 1, 4096)) --Changed
    net:add(relu(true))
    net:add(nn.BatchNormalization(4096))
    net:add(nn.Linear(4096, opt.embSize))
    net:add(nn.Normalize(2))


    return net
end


