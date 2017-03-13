local LiftedStructuredSimilaritySoftmaxCriterion, parent = torch.class('nn.LiftedStructuredSimilaritySoftmaxCriterion', 'nn.Criterion')

function LiftedStructuredSimilaritySoftmaxCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
end

require 'torchx'
function LiftedStructuredSimilaritySoftmaxCriterion:updateOutput(input, target)

    self.Li = torch.Tensor(input:size(1), 1):zero():type(torch.type(input))

    self.counter = 0
    for i = 1, input:size(1) do
        for j = 1, input:size(1) do
            if target[i] == target[j] and i ~= j then
                local indices = torch.find((target - target[i]):ne(0):type(torch.type(torch.FloatTensor())), 1)

                local dissValues = input[{ indices, {} }]
                local x1SubX3 = dissValues - input[i]:repeatTensor(dissValues:size(1), 1)
                local x2SubX3 = dissValues - input[j]:repeatTensor(dissValues:size(1), 1)

                local expX1SubX3 = torch.exp(self.alpha - x1SubX3:norm(2, 2))
                local expX2SubX3 = torch.exp(self.alpha - x2SubX3:norm(2, 2))

                self.counter = expX2SubX3:size(1) * 2

                self.Li[i] = (input[i] - input[j]):norm() + torch.log(expX1SubX3:sum() + expX2SubX3:sum())
            end
        end
    end
    self.output = torch.pow(self.Li, 2):sum() / (2 * self.counter)
    return self.output
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput(input, target)
    self.gradInput = torch.Tensor(input:size()):zero():type(torch.type(input))

    for i = 1, input:size(1) do
        for j = i+1, input:size(1) do
            if target[i] == target[j] and i < j then
                local subIJ = torch.csub(input[i], input[j])
                local normSubIJ = torch.norm(subIJ)
                for k = 1, input:size(1) do
                    if target[i] ~= target[k] then

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

    return self.gradInput
end
