--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 09/12/2016
-- Time: 22:33
-- To change this template use File | Settings | File Templates.
--



require 'loss/KLDivergence'
require 'loss/DoubleMargin'
require 'loss/ImprovedTriplet'
require 'loss/LSSS'
require 'loss/LMNNPull'
require 'loss/LMNNPush'
require 'loss/SoftPN'
require 'loss/Quadruplet'
require 'loss/Histogram'
require 'loss/GlobalTriplet'
require 'loss/GlobalSiamese'
require 'loss/Margin'

function selectCriterion()
    local criterion
    if opt.criterion == 't_orj' or opt.criterion == 't_entropy' then
        criterion = nn.TripletEmbeddingCriterion()
    elseif opt.criterion == 's_cosine' then
        criterion = nn.CosineEmbeddingCriterion(0.5)
    elseif opt.criterion == 's_hinge' then
        criterion = nn.HingeEmbeddingCriterion()
    elseif opt.criterion == 'crossentropy' then
        criterion = nn.CrossEntropyCriterion()
    elseif opt.criterion == 'kldiv' then
        criterion = nn.BatchKLDivCriterion(0.2)
    elseif opt.criterion == 'dist_ratio' then
        criterion = nn.DistanceRatioCriterion()
    elseif opt.criterion == 's_double_margin' then
        criterion = nn.DoubleMarginCriterion()
    elseif opt.criterion == 't_improved' then
        criterion = nn.ImprovedTripletCriterion()
    elseif opt.criterion == 'lsss' then
        criterion = nn.LiftedStructuredSimilaritySoftmaxCriterion()
    elseif opt.criterion == 'lmnn' then
        criterion = nn.LMNNPullCriterion()
    elseif opt.criterion == 'softPN' then
        criterion = nn.SoftPNCriterion()
    elseif opt.criterion == 'quadruplet' then
        criterion = nn.QuadrupletCriterion()
    elseif opt.criterion == 'histogram' then
        criterion = nn.HistogramCriterion()
    elseif opt.criterion == 't_global' then
        criterion = nn.TripletPlusGlobalCriterion()
    elseif opt.criterion == 's_global' then
        criterion = nn.SiamesePlusGlobalCriterion()
    elseif opt.criterion == 's_hadsell' then
        criterion = nn.HadsellMarginCriterion()
    elseif opt.criterion == 'margin' then
        criterion = nn.MultiMarginCriterion()
    elseif opt.criterion == 'multi' then
        criterion = nn.MultiCriterion()
        criterion:add(nn.MultiMarginCriterion(), 0.5):add(nn.CrossEntropyCriterion(), 0.5)
    end
    return criterion
end