--
-- Created by IntelliJ IDEA.
-- User: Cenk Bircanoglu
-- Date: 11/12/2016
-- Time: 16:22
-- To change this template use File | Settings | File Templates.
--

local CenterLossCriterion, parent = torch.class('nn.CenterLossCriterion', 'nn.Criterion')

function CenterLossCriterion:__init(learning_rate)
    parent.__init(self)
    self.learning_rate = learning_rate or 0.5
    self.means = nil
    self.gradInput = {}
    self.Li = nil
end


function CenterLossCriterion:updateOutput(inputs, labels)
    local N = inputs:size(1)

    local indeces = {}
    for i = 1, labels:size(1)
    do
        if indeces[labels[i]] == nil then
            indeces[labels[i]] = { i }
        else
            table.insert(indeces[labels[i]], i)
        end
    end
    local means_table = {}
    for i, line in ipairs(indeces) do
        means_table[i] = inputs[{ line, {} }]:mean(1)
        assert(means_table[i]:dim() == inputs:dim())
    end



    self.Li = torch.Tensor(N):fill(0)
    if self.means == nil then
        self.means = torch.Tensor(inputs:size()):zero()
    end
    for i = 1, labels:size(1)
    do
        local zero_means = self.means[{ { i }, {} }]:clone():zero()
        if torch.all(torch.eq(self.means[{ { i }, {} }], zero_means)) then
            self.means[{ { i }, {} }] = means_table[labels[i]]
        else
            self.means[{ { i }, {} }] = (self.learning_rate * self.means[{ { i }, {} }] + means_table[labels[i]]) / 2
        end
    end

    self.Li = (inputs - self.means):norm(2, 2) / 2
    self.output = self.Li:sum() / N
    return self.output
end

function CenterLossCriterion:updateGradInput(inputs, labels)
    self.gradInput = torch.Tensor(inputs:size(1), inputs:size(2)):fill(0)
    self.gradInput = inputs - self.means
    return self.gradInput
end
