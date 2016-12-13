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
    elseif opt.criterion == 'mse' then
        criterion = nn.MSECriterion()
    elseif opt.criterion == 'mmc' then
        criterion = nn.MultiMarginCriterion()
    elseif opt.criterion == 'meanLoss' then
        criterion = nn.MeanLossCriterion(opt.alpha, opt.imagesPerPerson)
    elseif opt.criterion == 'centerLoss' then
        criterion = nn.CenterLossCriterion(opt.alpha, opt.imagesPerPerson)
    elseif opt.criterion == 'minDiff' then
        criterion = nn.MinDiffLossCriterion(opt.alpha, opt.imagesPerPerson)
    elseif opt.criterion == 'cons' then
        criterion = nn.ConsLossCriterion(opt.alpha, opt.imagesPerPerson)
    end
    return criterion
end