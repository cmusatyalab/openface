local LiftedStructuredSimilaritySoftmaxCriterion, parent = torch.class('nn.LiftedStructuredSimilaritySoftmaxCriterion', 'nn.Criterion')
require 'torchx'
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

                local dissValues = input[{ torch.find((target - target[i]):ne(0), 1), {} }]
                local x1SubX3 = dissValues - input[i]:repeatTensor(dissValues:size(1), 1)
                local x2SubX3 = dissValues - input[j]:repeatTensor(dissValues:size(1), 1)
                local expX1SubX3 = torch.exp(self.alpha - torch.norm(x1SubX3, 2, 2))
                local expX2SubX3 = torch.exp(self.alpha - torch.norm(x2SubX3, 2, 2))

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
    --aa = self:updateGradInput1(input, target)
    for i = 1, input:size(1) do
        local dissValues = input[{ torch.find((target - target[i]):ne(0), 1), {} }]
        local x1SubX3 = dissValues - input[i]:repeatTensor(dissValues:size(1), 1)
        local normX1SubX3 = torch.norm(x1SubX3, 2, 2)

        for j = 1, input:size(1) do
            if target[i] == target[j] and i < j then
                local subIJ = torch.csub(input[i], input[j])
                local normSubIJ = torch.norm(subIJ)

                local x2SubX3 = dissValues - input[j]:repeatTensor(dissValues:size(1), 1)
                local normX2SubX3 = torch.norm(x2SubX3, 2, 2)

                local dividedIK = -torch.exp(self.alpha - normX1SubX3)
                local dividedJK = -torch.exp(self.alpha - normX2SubX3)

                local dividing = torch.exp(self.Li[i] - normSubIJ)[1]

                local firstDivIK = dividedIK / dividing
                local firstDivIJ = dividedJK / dividing

                self.gradInput[{ torch.find((target - target[i]):ne(0), 1), {} }] = self.gradInput[{ torch.find((target - target[i]):ne(0), 1), {} }]
                        + torch.cmul(torch.cdiv(x1SubX3, normX1SubX3:repeatTensor(1, x1SubX3:size(2))), firstDivIK:repeatTensor(1, x1SubX3:size(2)))
                        + torch.cmul(torch.cdiv(x2SubX3, normX2SubX3:repeatTensor(1, x2SubX3:size(2))), firstDivIJ:repeatTensor(1, x2SubX3:size(2)))

                self.gradInput[i] = self.gradInput[i] + (subIJ / normSubIJ)
                        + -torch.cmul(torch.cdiv(x1SubX3, normX1SubX3:repeatTensor(1, x1SubX3:size(2))), firstDivIK:repeatTensor(1, x1SubX3:size(2))):sum(1)

                self.gradInput[j] = self.gradInput[j] + (-subIJ / normSubIJ)
                        + -torch.cmul(torch.cdiv(x2SubX3, normX2SubX3:repeatTensor(1, x2SubX3:size(2))), firstDivIJ:repeatTensor(1, x2SubX3:size(2))):sum(1)
            end
        end
    end
    self.gradInput = torch.cmul(self.Li:expandAs(input), self.gradInput) / self.counter
    print(aa)
    return self.gradInput
end


function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput1(input, target)
    self.gradInput1 = torch.Tensor(input:size()):zero():type(torch.type(input))

    for i = 1, input:size(1) do
        for j = 1, input:size(1) do
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

                        self.gradInput1[i] = self.gradInput1[i] + (subIK * (dividedIK / (dividing * normSubIK)))
                        self.gradInput1[k] = self.gradInput1[k] + -(subIK * (dividedIK / (dividing * normSubIK)))

                        self.gradInput1[j] = self.gradInput1[j] + (subJK * (dividedJK / (dividing * normSubJK)))
                        self.gradInput1[k] = self.gradInput1[k] + -(subJK * (dividedJK / (dividing * normSubJK)))
                    end
                end
                self.gradInput1[i] = self.gradInput1[i] + (subIJ / normSubIJ)

                self.gradInput1[j] = self.gradInput1[j] + (-subIJ / normSubIJ)
            end
        end
    end

    self.gradInput1 = torch.cmul(self.Li:expandAs(input), self.gradInput1) / self.counter

    return self.gradInput1
end
