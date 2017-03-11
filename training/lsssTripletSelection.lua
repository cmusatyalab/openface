--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 21:40
-- To change this template use File | Settings | File Templates.
--


function LSSSTriplets(embeddings, numImages, numPerClass)
    local as_table = {}
    local ps_table = {}
    local ns_table = {}

    local mapper = {}

    for i = 1, embeddings:size(1) do
        local classIdi = findClassId(i, numPerClass)
        for j = 1, embeddings:size(1) do
            if i ~= j then
                local classIdj = findClassId(j, numPerClass)
                if classIdj == classIdi then
                    for k = 1, embeddings:size(1) do
                        local classIdk = findClassId(k, numPerClass)
                        if classIdj ~= classIdk then
                            table.insert(as_table, embeddings[i])
                            table.insert(ps_table, embeddings[j])
                            table.insert(ns_table, embeddings[k])

                            table.insert(mapper, { i, j, k })
                        end
                    end
                end
            end
        end
    end

    local as = torch.concat(as_table):view(table.getn(as_table), opt.embSize)
    local ps = torch.concat(ps_table):view(table.getn(ps_table), opt.embSize)
    local ns = torch.concat(ns_table):view(table.getn(ns_table), opt.embSize)

    local apn
    if opt.cuda then
        local asCuda = torch.CudaTensor()
        local psCuda = torch.CudaTensor()
        local nsCuda = torch.CudaTensor()

        local sz = as:size()
        asCuda:resize(sz):copy(as)
        psCuda:resize(sz):copy(ps)
        nsCuda:resize(sz):copy(ns)

        apn = { asCuda, psCuda, nsCuda }
    else
        apn = { as, ps, ns }
    end
    print(table.getn(mapper))
    return apn, mapper
end

