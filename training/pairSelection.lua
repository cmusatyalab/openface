--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 13/12/2016
-- Time: 10:28
-- To change this template use File | Settings | File Templates.
--



function pairss(embeddings, numPerClass, simi, dissimi)
    local a1s_table = {}
    local a2s_table = {}
    local targets_table = {}
    local mapper = {}
    for i = 1, embeddings:size(1) do
        local classIdi = findClassId(i, numPerClass)
        for j = 1, embeddings:size(1) do
            local classIdj = findClassId(j, numPerClass)
            table.insert(a1s_table, embeddings[i])
            table.insert(a2s_table, embeddings[j])

            local target
            if classIdi == classIdj then
                target = simi
            else
                target = dissimi
            end
            table.insert(targets_table, target)
            table.insert(mapper, { i, j })
        end
    end


    local a1s = torch.concat(a1s_table):view(table.getn(a1s_table), opt.embSize)
    local a2s = torch.concat(a2s_table):view(table.getn(a2s_table), opt.embSize)
    local targets = torch.Tensor(targets_table)


    local as
    if opt.cuda then
        targets = targets:cuda()

        as = { a1s:cuda(), a2s:cuda() }
    else
        as = { a1s, a2s }
    end

    return as, targets, mapper
end

