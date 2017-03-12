local LiftedStructuredSimilaritySoftmaxCriterion, parent = torch.class('nn.LiftedStructuredSimilaritySoftmaxCriterion', 'nn.Criterion')
local epsilon = 1e-35
function LiftedStructuredSimilaritySoftmaxCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateOutput(input, target)

    self.Li = torch.Tensor(input:size(1), 1):zero():type(torch.type(input))

    self.counter = 0
    for i = 1, input:size(1) do
        for j = 1, input:size(1) do
            if target[i] == target[j] and i ~= j then
                self.counter = self.counter + 1
                local total1 = 0
                local total2 = 0
                for k = 1, input:size(1) do

                    if target[i] ~= target[k] then
                        total1 = total1 + torch.exp(self.alpha - (input[i] - input[k]):norm())

                        total2 = total2 + torch.exp(self.alpha - (input[j] - input[k]):norm())
                    end
                end

                self.Li[i] = (input[i] - input[j]):norm() + torch.log(total1 + total2)
            end
        end
    end

    self.output = torch.pow(self.Li, 2):sum() / (2 * self.counter)
    return self.output
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput(input, target)
    self.gradInput = torch.Tensor(input:size()):zero():type(torch.type(input))

    for i = 1, input:size(1) do
        for j = 1, input:size(1) do
            if target[i] == target[j] and i < j then
                local subIJ = torch.csub(input[i], input[j])
                local normSubIJ = torch.norm(subIJ)
                for k = 1, input:size(1) do
                    if target[i] ~= target[k] then
                        --print(i, j, k)
                        local subIK = torch.csub(input[i], input[k])
                        local normSubIK = torch.norm(subIK)

                        local subJK = torch.csub(input[j], input[k])
                        local normSubJK = torch.norm(subJK)

                        local dividedIK = -torch.exp(self.alpha - normSubIK)
                        local dividedJK = -torch.exp(self.alpha - normSubJK)

                        local dividing = torch.exp(self.Li[i][1] - normSubIJ)

                        self.gradInput[i] = self.gradInput[i] + (subIK * (dividedIK / (dividing * normSubIK)))
                        self.gradInput[k] = self.gradInput[k] + -(subIK * (dividedIK / (dividing * normSubIK)))

                        self.gradInput[i] = self.gradInput[j] + (subJK * (dividedJK / (dividing * normSubJK)))
                        self.gradInput[k] = self.gradInput[k] + -(subJK * (dividedJK / (dividing * normSubJK)))
                    end
                end
                self.gradInput[i] = self.gradInput[i] + (subIJ / normSubIJ)

                self.gradInput[j] = self.gradInput[j] + (-subIJ / normSubIJ)
            end
        end
    end

    self.gradInput = torch.cmul(self.Li:expandAs(input), self.gradInput) / self.counter
    --print(self.gradInput)
    return self.gradInput
end
