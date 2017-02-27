--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 22:33
-- To change this template use File | Settings | File Templates.
--


require 'loss/CenterLoss'
require 'loss/batchkl'
function selectCriterion()
    local criterion
    if opt.criterion == 'classification' then
        criterion = nn.ClassNLLCriterion()
    elseif opt.criterion == 'triplet' then
        criterion = nn.TripletEmbeddingCriterion(opt.alpha)
    elseif opt.criterion == 'siamese' then
        criterion = nn.CosineEmbeddingCriterion()
    elseif opt.criterion == 'center' then
        criterion = nn.MultiCriterion():add(nn.ClassNLLCriterion()):add(nn.CenterLossCriterion(), 0.003)
    elseif opt.criterion == 'kldiv' then
        criterion = nn.BatchKLDivCriterion()
    end
    return criterion
end