--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 13/12/2016
-- Time: 10:28
-- To change this template use File | Settings | File Templates.
--



function pairss(embeddings, numPerClass)
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
            local target = 1
            if classIdi ~= classIdj then
                target = -1
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
        local a1sCuda = torch.CudaTensor()
        local a2sCuda = torch.CudaTensor()
        local targetsCuda = torch.CudaTensor()

        local sz = as:size()
        a1sCuda:resize(sz):copy(a1s)
        a2sCuda:resize(sz):copy(a2s)
        targetsCuda:resize(sz):copy(targets)

        as = { a1s, a2s }
    else
        as = { a1s, a2s }
    end

    return as, targets, mapper
end

