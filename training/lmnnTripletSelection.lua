--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 21:40
-- To change this template use File | Settings | File Templates.
--


function LMNNTriplets(embeddings, numImages, numPerClass)
    local as_table = {}
    local ps_table = {}
    local ns_table = {}
    local triplet_idx = {}

    local x1_table = {}
    local x2_table = {}
    local pair_idx = {}
    local num_example_per_idx = torch.Tensor(embeddings:size(1))
    num_example_per_idx:zero()

    local tripIdx = 1
    local embStartIdx = 1
    local numTrips = 0
    for i = 1, opt.peoplePerBatch do
        local n = numPerClass[i]
        for j = 1, n - 1 do -- For every image in the batch.
            local aIdx = embStartIdx + j - 1

            local diff = embeddings - embeddings[{ { aIdx } }]:expandAs(embeddings)
            local norms = diff:norm(2, 2):pow(2):squeeze()
            for pair = j, n - 1 do -- For every possible positive pair.
                local pIdx = embStartIdx + pair
                local fff = (embeddings[aIdx] - embeddings[pIdx]):norm(2)
                local normsP = norms - torch.Tensor(embeddings:size(1)):fill(fff * fff)

                -- Set the indices of the same class to the max so they are ignored.
                normsP[{ { embStartIdx, embStartIdx + n - 1 } }] = normsP:max()

                -- Get indices of images within the margin.
                local in_margin = normsP:lt(opt.alpha)
                local allNeg = torch.find(in_margin, 1)

                -- Use only non-random triplets.
                -- Random triples (which are beyond the margin) will just produce gradient = 0,
                -- so the average gradient will decrease.
                if table.getn(allNeg) ~= 0 then
                    selNegIdx = allNeg[math.random(table.getn(allNeg))]
                    -- Add the embeding of each example.
                    table.insert(as_table, embeddings[aIdx])
                    table.insert(ps_table, embeddings[pIdx])
                    table.insert(ns_table, embeddings[selNegIdx])
                    -- Add the original index of triplets.
                    table.insert(triplet_idx, { aIdx, pIdx, selNegIdx })
                    -- Increase the number of times of using each example.
                    num_example_per_idx[aIdx] = num_example_per_idx[aIdx] + 1
                    num_example_per_idx[pIdx] = num_example_per_idx[pIdx] + 1
                    num_example_per_idx[selNegIdx] = num_example_per_idx[selNegIdx] + 1
                    tripIdx = tripIdx + 1
                end

                numTrips = numTrips + 1
            end
        end
        embStartIdx = embStartIdx + n
    end

    for i = 1, embeddings:size(1) do
        local classIdi = findClassId(i, numPerClass[1])
        for j = 1, embeddings:size(1) do
            if i ~= j then
                local classIdj = findClassId(j, numPerClass[1])

                local target
                if classIdi == classIdj then
                    table.insert(x1_table, embeddings[i])
                    table.insert(x2_table, embeddings[j])
                    table.insert(pair_idx, { i, j })
                end
            end
        end
    end
    assert(embStartIdx - 1 == numImages)
    local nTripsFound = table.getn(as_table)
    print(('  + (nTrips, nTripsFound) = (%d, %d)'):format(numTrips, nTripsFound))

    if nTripsFound == 0 then
        print("Warning: nTripsFound == 0. Skipping batch.")
        return nil, nil
    end

    local as = torch.concat(as_table):view(table.getn(as_table), opt.embSize)
    local ps = torch.concat(ps_table):view(table.getn(ps_table), opt.embSize)
    local ns = torch.concat(ns_table):view(table.getn(ns_table), opt.embSize)
    local x1 = torch.concat(x1_table):view(table.getn(x1_table), opt.embSize)
    local x2 = torch.concat(x2_table):view(table.getn(x2_table), opt.embSize)

    local apn_x
    if opt.cuda then
        local asCuda = torch.CudaTensor()
        local psCuda = torch.CudaTensor()
        local nsCuda = torch.CudaTensor()
        local x1Cuda = torch.CudaTensor()
        local x2Cuda = torch.CudaTensor()

        local sz = as:size()
        asCuda:resize(sz):copy(as)
        psCuda:resize(sz):copy(ps)
        nsCuda:resize(sz):copy(ns)
        sz = x1:size()

        x1Cuda:resize(sz):copy(x1)
        x2Cuda:resize(sz):copy(x2)

        apn_x = { { asCuda, psCuda, nsCuda }, { x1Cuda, x2Cuda } }
    else
        apn_x = { { as, ps, ns }, { x1, x2 } }
    end

    return apn_x, { triplet_idx, pair_idx }
end