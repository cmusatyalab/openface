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
require 'loss/LMNN'
require 'loss/SoftPN'
require 'loss/Quadruplet'
require 'loss/Histogram'
require 'loss/GlobalTriplet'
require 'loss/GlobalSiamese'

function selectCriterion()
    local criterion
    if opt.criterion == 't_orj' then
        criterion = nn.TripletEmbeddingCriterion()
    elseif opt.criterion == 's_cosine' then
        criterion = nn.CosineEmbeddingCriterion(0.5)
    elseif opt.criterion == 's_hinge' then
        criterion = nn.HingeEmbeddingCriterion()
    elseif opt.criterion == 'crossentropy' then
        criterion = nn.CrossEntropyCriterion()
    elseif opt.criterion == 'kldiv' then
        criterion = nn.BatchKLDivCriterion(0.005)
    elseif opt.criterion == 'dist_ratio' then
        criterion = nn.DistanceRatioCriterion()
    elseif opt.criterion == 's_double_margin' then
        criterion = nn.DoubleMarginCriterion()
    elseif opt.criterion == 't_improved' then
        criterion = nn.ImprovedTripletCriterion()
    elseif opt.criterion == 'lsss' then
        criterion = nn.LiftedStructuredSimilaritySoftmaxCriterion()
    elseif opt.criterion == 'lmnn' then
        criterion = nn.LargeMarginNearestNeighborCriterion()
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
    end
    return criterion
end