local LiftedStructuredSimilaritySoftmaxCriterion, parent = torch.class('nn.LiftedStructuredSimilaritySoftmaxCriterion', 'nn.Criterion')

function LiftedStructuredSimilaritySoftmaxCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
    self.counter = {}
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateOutput(input, values)
    local x1 = input[1]
    local x2 = input[2]
    local target = values[1]
    local mapper = values[2]

    local diffs = (x1 - x2):norm(2, 2):cmul(target) + torch.exp(self.alpha - (x1 - x2):norm(2, 2)):cmul(1 - target)
    self.diffs = diffs
    self.li = torch.Tensor(table.getn(mapper), 1):zero()
    local poscount = 0
    for i = 1, table.getn(mapper) do
        self.counter[mapper[i][1]] = 0
        if target[i] == 1 then
            poscount = poscount + 1
            local total1 = 0
            local total2 = 0
            for j = 1, table.getn(mapper) do
                if target[j] == 0 then
                    if mapper[i][1] == mapper[j][1] then
                        total1 = total1 + diffs[j]
                    end
                    if mapper[i][2] == mapper[j][1] then
                        total2 = total2 + diffs[j]
                    end
                end
            end
            self.li[i] = diffs[i] + torch.log(total1 + total2)
        end
    end
    self.poscount = poscount
    self.Li = torch.max(torch.cat(torch.Tensor(self.li:size(1)):zero():type(torch.type(self.li)), self.li, 2), 2):pow(2)
    self.output = self.Li:sum() / (2 * poscount)
    return self.output
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput(input, values)
    local x1 = input[1]
    local x2 = input[2]
    local target = values[1]
    local mapper = values[2]
    local diff1 = (x1 - x2):cmul((x1 - x2):norm(2, 2):expandAs(x1))
    local diff2 = -torch.exp(self.alpha - (x1 - x2):norm(2, 2)):cdiv(torch.exp(self.li - (x1 - x2):norm(2, 2)))

    self.gradInput = torch.Tensor(table.getn(self.counter), x1:size(2)):type(x1:type()):zero()
    for i = 1, table.getn(mapper) do
        if target[i] == 1 then

            self.gradInput[mapper[i][1]] = diff1[i]:view(x1:size(2), 1) * (self.Li[i] / self.poscount)
            self.gradInput[mapper[i][2]] = diff1[i]:view(x1:size(2), 1) * (-self.Li[i] / self.poscount)
            for j = 1, table.getn(mapper) do
                if target[j] == 0 then
                    if mapper[i][1] == mapper[j][1] then
                        self.gradInput[mapper[i][1]]:add(diff1[j]:view(x1:size(2), 1) * ((self.Li[j] / self.poscount) * -diff2[j]))
                        self.gradInput[mapper[j][1]]:add(diff1[j]:view(x1:size(2), 1) * ((self.Li[j] / self.poscount) * diff2[j]))
                    end
                    if mapper[i][2] == mapper[j][1] then
                        self.gradInput[mapper[i][2]]:add(diff1[j]:view(x1:size(2), 1) * ((self.Li[j] / self.poscount) * -diff2[j]))
                        self.gradInput[mapper[j][2]]:add(diff1[j]:view(x1:size(2), 1) * ((self.Li[j] / self.poscount) * diff2[j]))
                    end
                end
            end
        end
    end

    return self.gradInput
end
