--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 22:33
-- To change this template use File | Settings | File Templates.
--



function selectCriterion()
    local criterion
    if opt.criterion == 'contrastive' then
        criterion = nn.ClassNLLCriterion()
    elseif opt.criterion == 'triplet' then
        criterion = nn.TripletEmbeddingCriterion(opt.alpha)
    elseif opt.criterion == 'siamese' then
        criterion = nn.CosineEmbeddingCriterion()
    end
    return criterion
end