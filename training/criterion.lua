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
    end
    return criterion
end