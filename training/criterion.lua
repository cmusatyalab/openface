--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 22:33
-- To change this template use File | Settings | File Templates.
--



function selectCriterion()
    local criterion = nn.TripletEmbeddingCriterion(opt.alpha)
    if opt.criterion == 'loglikelihood' then
        criterion = nn.ClassNLLCriterion()
    elseif opt.criterion == 'cosine' then
        criterion = nn.CosineEmbeddingCriterion(opt.alpha)
    elseif opt.criterion == 'l1hinge' then
        criterion = nn.L1HingeEmbeddingCriterion(opt.alpha)
    elseif opt.criterion == 'marginranking' then
        criterion = nn.MarginRankingCriterion(opt.alpha)
    end
    return criterion
end