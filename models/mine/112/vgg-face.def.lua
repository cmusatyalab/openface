--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 14.01.2017
-- Time: 09:49
-- To change this template use File | Settings | File Templates.
--

imgDim = 112


function createModel()

    local conv = nn.SpatialConvolutionMM
    local relu = nn.ReLU
    local mp = nn.SpatialMaxPooling
    local net = nn.Sequential()

    net:add(conv(3, 64, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(conv(64, 64, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(mp(2, 2, 2, 2))
    net:add(conv(64, 128, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(conv(128, 128, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(mp(2, 2, 2, 2))
    net:add(conv(128, 256, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(conv(256, 256, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(conv(256, 256, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(mp(2, 2, 2, 2))
    net:add(conv(256, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(mp(2, 2, 2, 2))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(conv(512, 512, 3, 3, 1, 1, 1, 1))
    net:add(relu(true))
    net:add(mp(2, 2, 2, 2))

    -- Validate shape with:
    -- net:add(nn.Reshape(25088))

    net:add(nn.View(4608))
    net:add(nn.Linear(4608, 4096))
    net:add(relu(true))

    net:add(nn.Linear(4096, opt.embSize))
    net:add(nn.Normalize(2))


    return net
end


