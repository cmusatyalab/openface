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
        for j = 1, input:size(1) do
            if target[i] == target[j] and i < j then
                local subIJ = torch.csub(input[i], input[j])
                local normSubIJ = torch.norm(subIJ)
                local indices = torch.find((target - target[i]):ne(0):type(torch.type(torch.FloatTensor())), 1)
                local dissValues = input[{ indices, {} }]
                local x1SubX3 = dissValues - input[i]:repeatTensor(dissValues:size(1), 1)
                local normX1SubX3 = torch.norm(x1SubX3)
                local x2SubX3 = dissValues - input[j]:repeatTensor(dissValues:size(1), 1)
                local normX2SubX3 = torch.norm(x2SubX3)

                local dividedIK = -torch.exp(self.alpha - normX1SubX3)
                local dividedJK = -torch.exp(self.alpha - normX2SubX3)

                local dividing = torch.exp(self.Li[i] - normSubIJ)[1]

                self.gradInput[{ indices, {} }] = self.gradInput[{ indices, {} }]
                        + -(x1SubX3:sum(1) * (dividedIK / (dividing * normX1SubX3))):repeatTensor(dissValues:size(1), 1)
                        + -(x2SubX3:sum(1) * (dividedJK / (dividing * normX2SubX3))):repeatTensor(dissValues:size(1), 1)

                self.gradInput[i] = self.gradInput[i] + (subIJ / normSubIJ) + (x1SubX3:sum(1) * (dividedIK / (dividing * normX1SubX3)))
                self.gradInput[j] = self.gradInput[j] + (-subIJ / normSubIJ) + (x2SubX3:sum(1) * (dividedJK / (dividing * normX2SubX3)))
            end
        end
    end

    self.gradInput = torch.cmul(self.Li:expandAs(input), self.gradInput) / self.counter

    return self.gradInput
end
