--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 21:40
-- To change this template use File | Settings | File Templates.
--


function fullTriplets(embeddings, numImages, numPerClass)
    local a1s_table = {}
    local a2s_table = {}
    local a3s_table = {}
    local targets_table = {}
    local mapper = {}
    for i = 1, embeddings:size(1) do
        for j = 1, embeddings:size(1) do
            local classIdj = findClassId(j, numPerClass)
            if i ~= j then
                for k = 1, embeddings:size(1) do
                    local classIdk = findClassId(k, numPerClass)
                    table.insert(a1s_table, embeddings[i])
                    table.insert(a2s_table, embeddings[j])
                    table.insert(a3s_table, embeddings[k])

                    local target
                    if classIdj == classIdk then
                        target = 1
                    else
                        target = 0
                    end
                    table.insert(targets_table, target)
                    table.insert(mapper, { i, j, k })
                end
            end
        end
    end


    local a1s = torch.concat(a1s_table):view(table.getn(a1s_table), opt.embSize)
    local a2s = torch.concat(a2s_table):view(table.getn(a2s_table), opt.embSize)
    local a3s = torch.concat(a3s_table):view(table.getn(a3s_table), opt.embSize)
    local targets = torch.Tensor(targets_table)

    local nTripsFound = table.getn(a1s_table)
    print('Triplets found', nTripsFound)
    local as
    if opt.cuda then
        targets = targets:cuda()

        as = { a1s:cuda(), a2s:cuda(), a3s:cuda() }
    else
        as = { a1s, a2s, a3s }
    end

    return as, targets, mapper
end

