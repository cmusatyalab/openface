--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 07:59
-- To change this template use File | Settings | File Templates.
--
imgDim = 64

function createModel()
    local net = nn.Sequential()

    --net:add(nn.SpatialConvolutionMM(3, 64, 7, 7, 2, 2, 3, 3))
    net:add(nn.SpatialConvolutionMM(3, 64, 5, 5, 2, 2, 3, 3))
    net:add(nn.SpatialBatchNormalization(64))
    net:add(nn.ReLU())

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
    net:add(nn.Inception {
        inputSize = 192,
        kernelSize = { 3, 5 },
        kernelStride = { 1, 1 },
        outputSize = { 128, 32 },
        reduceSize = { 96, 16, 32, 64 },
        pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
        batchNorm = true
    })

    -- Inception (3b)
    net:add(nn.Inception {
        inputSize = 256,
        kernelSize = { 3, 5 },
        kernelStride = { 1, 1 },
        outputSize = { 128, 64 },
        reduceSize = { 96, 32, 64, 64 },
        pool = nn.SpatialLPPooling(256, 2, 3, 3, 1, 1),
        batchNorm = true
    })

    -- Inception (3c)
    net:add(nn.Inception {
        inputSize = 320,
        kernelSize = { 3, 5 },
        kernelStride = { 2, 2 },
        outputSize = { 256, 64 },
        reduceSize = { 128, 32, nil, nil },
        pool = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1),
        batchNorm = true
    })

    -- Inception (4a)
    net:add(nn.Inception {
        inputSize = 640,
        kernelSize = { 3, 5 },
        kernelStride = { 1, 1 },
        outputSize = { 192, 64 },
        reduceSize = { 96, 32, 128, 256 },
        pool = nn.SpatialLPPooling(640, 2, 3, 3, 1, 1),
        batchNorm = true
    })

    -- Inception (4b)
    net:add(nn.Inception {
        inputSize = 640,
        kernelSize = { 3, 5 },
        kernelStride = { 1, 1 },
        outputSize = { 224, 64 },
        reduceSize = { 112, 32, 128, 224 },
        pool = nn.SpatialLPPooling(640, 2, 3, 3, 1, 1),
        batchNorm = true
    })

    -- Inception (4c)
    net:add(nn.Inception {
        inputSize = 640,
        kernelSize = { 3, 5 },
        kernelStride = { 1, 1 },
        outputSize = { 256, 64 },
        reduceSize = { 128, 32, 128, 192 },
        pool = nn.SpatialLPPooling(640, 2, 3, 3, 1, 1),
        batchNorm = true
    })

    -- Inception (4d)
    net:add(nn.Inception {
        inputSize = 640,
        kernelSize = { 3, 5 },
        kernelStride = { 1, 1 },
        outputSize = { 288, 64 },
        reduceSize = { 144, 32, 128, 160 },
        pool = nn.SpatialLPPooling(640, 2, 3, 3, 1, 1),
        batchNorm = true
    })

    -- Inception (4e)
    net:add(nn.Inception {
        inputSize = 640,
        kernelSize = { 3, 5 },
        kernelStride = { 2, 2 },
        outputSize = { 256, 128 },
        reduceSize = { 160, 64, nil, nil },
        pool = nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1),
        batchNorm = true
    })

    -- Inception (5a)
    net:add(nn.Inception {
        inputSize = 1024,
        kernelSize = { 3, 5 },
        kernelStride = { 1, 1 },
        outputSize = { 384, 128 },
        reduceSize = { 192, 48, 128, 384 },
        pool = nn.SpatialLPPooling(960, 2, 3, 3, 1, 1),
        batchNorm = true
    })

    -- Inception (5b)
    net:add(nn.Inception {
        inputSize = 1024,
        kernelSize = { 3, 5 },
        kernelStride = { 1, 1 },
        outputSize = { 384, 128 },
        reduceSize = { 192, 48, 128, 384 },
        pool = nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1),
        batchNorm = true
    })

    --net:add(nn.SpatialAveragePooling(7, 7))
    net:add(nn.SpatialAveragePooling(3, 3))


    net:add(nn.View(1024))
    net:add(nn.Linear(1024, opt.embSize))
    net:add(nn.Normalize(2))

    return net
end
