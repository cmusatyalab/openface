local HistogramCriterion, parent = torch.class('nn.HistogramCriterion', 'nn.Criterion')
require 'torchx'
function HistogramCriterion:__init(alpha, gridDelta)
    parent.__init(self)
    self.alpha = alpha or 0.2
    self.Li = torch.Tensor()
    self.gradInput = {}
    self.gridDelta = gridDelta or 0.01
end

function HistogramCriterion:getGrid(input)
    self.grid = torch.linspace(-1, 1, (2 / self.gridDelta) + 1):type(torch.type(input))

end

function HistogramCriterion:createMapping(input)
    local places_to_bins = {}
    local bins_to_places = {}
    for i = 1, self.grid:size(1) do
        bins_to_places[i] = {}
    end
    for i = 1, input:size(1) do
        for j = 1, self.grid:size(1) - 1 do
            if input[i] >= self.grid[j] and input[i] <= self.grid[j + 1] then
                table.insert(bins_to_places[j], i)
                places_to_bins[i] = j
            end
        end
    end
    return places_to_bins, bins_to_places
end

local function getDistributionDensity(x, bins_to_places, grid, gridDelta)
    local p = torch.Tensor(grid:size(1)):zero():type(torch.type(grid))

    for i = 1, grid:size(1) do
        local left_add = 0
        if i > 1 then
            for j = 1, table.getn(bins_to_places[i - 1]) do
                left_add = left_add + (x[bins_to_places[i - 1][j]] - grid[i - 1])
            end
        end
        local right_add = 0
        if i < grid:size(1) then

            for j = 1, table.getn(bins_to_places[i]) do
                right_add = right_add + (grid[i + 1] - x[bins_to_places[i][j]])
            end
        end
        p[i] = left_add + right_add
    end

    p = p / (x:size(1) * gridDelta)
    return p
end

function HistogramCriterion:getL()

    local L = torch.Tensor(self.grid:size(1), self.grid:size(1)):fill(1):type(torch.type(self.grid))
    for i = 1, self.grid:size(1) do

        L[i] = (self.grid[i] - self.grid):le(0):type(torch.type(self.grid))
    end
    return L
end

function HistogramCriterion:updateOutput(input, target)
    self:getGrid(input)
    self.pos_indices = torch.find((target):eq(1):type(torch.type(torch.FloatTensor())), 1)
    self.neg_indices = torch.find((target):eq(-1):type(torch.type(torch.FloatTensor())), 1)

    self.pos = input:index(1, torch.LongTensor(self.pos_indices))
    self.neg = input:index(1, torch.LongTensor(self.neg_indices))

    self.places_to_bins_pos, self.bins_to_places_pos = self:createMapping(self.pos)
    self.places_to_bins_neg, self.bins_to_places_neg = self:createMapping(self.neg)

    self.distr_pos = getDistributionDensity(self.pos, self.bins_to_places_pos, self.grid, self.gridDelta)
    self.distr_neg = getDistributionDensity(self.neg, self.bins_to_places_neg, self.grid, self.gridDelta)

    self.L = self:getL()
    self.output = (self.distr_pos:view(self.distr_pos:size(1), 1):t() * self.L) * self.distr_neg

    return self.output[1]
end

function HistogramCriterion:calculateLossGradOverDistribution()

    local grad_pos = self.L * self.distr_neg:view(self.distr_neg:size(1), 1)
    local grad_neg = self.distr_pos:view(self.distr_pos:size(1), 1):t() * self.L
    return grad_pos, grad_neg
end

function HistogramCriterion:calculateLossGradOverBinsForHist(grad_pos, grad_neg)
    local gradPos = grad_pos:clone()
    local gradNeg = grad_neg:clone()
    for i = 2, grad_pos:size(1) do
        gradPos[i] = grad_pos[i] - grad_pos[i - 1]
    end
    gradPos = gradPos / (self.gridDelta * self.pos:size(1))
    for i = 2, grad_neg:size(1) do
        gradNeg[i] = grad_neg[i] - grad_neg[i - 1]
    end
    gradNeg = gradNeg / (self.gridDelta * self.neg:size(1))

    return gradPos, gradNeg
end

function HistogramCriterion:getGradOverData(data, grad_over_bins, places_to_bins)
    local grad = torch.Tensor(data:size()):zero():type(torch.type(data))
    for i = 1, data:size(1) do
        grad[i] = grad_over_bins[places_to_bins[i] + 1]
    end
    return grad
end

function HistogramCriterion:updateGradInput(input, target)
    local grad_pos_distr, grad_neg_distr = self:calculateLossGradOverDistribution()
    grad_pos_distr = grad_pos_distr:view(grad_pos_distr:size(1))
    grad_neg_distr = grad_neg_distr:view(grad_neg_distr:size(2))
    local grad_pos_bin, grad_neg_bin = self:calculateLossGradOverBinsForHist(grad_pos_distr, grad_neg_distr)


    local grad_pos = self:getGradOverData(self.pos, grad_pos_bin, self.places_to_bins_pos)
    local grad_neg = self:getGradOverData(self.neg, grad_neg_bin, self.places_to_bins_neg)
    self.gradInput = torch.Tensor(input:size()):type(torch.type(input))
    self.gradInput:indexCopy(1, torch.LongTensor(self.pos_indices), grad_pos)
    self.gradInput:indexCopy(1, torch.LongTensor(self.neg_indices), grad_neg)
    return self.gradInput
end
