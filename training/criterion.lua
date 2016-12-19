--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 22:33
-- To change this template use File | Settings | File Templates.
--



function selectCriterion()
    local criterion = nil
    if opt.criterion == 'loglikelihood' then
        criterion = nn.ClassNLLCriterion()
    elseif opt.criterion == 'cosine' then
        criterion = nn.CosineEmbeddingCriterion(0.8)
    elseif opt.criterion == 'l1hinge' then
        criterion = nn.L1HingeEmbeddingCriterion()
    elseif opt.criterion == 'triplet' then
        criterion = nn.TripletEmbeddingCriterion(opt.alpha)
    end
    return criterion
end