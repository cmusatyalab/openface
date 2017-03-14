local LiftedStructuredSimilaritySoftmaxCriterion, parent = torch.class('nn.LiftedStructuredSimilaritySoftmaxCriterion', 'nn.Criterion')
require 'torchx'
function LiftedStructuredSimilaritySoftmaxCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateOutput(input, target)

    self.Li = torch.Tensor(input:size(1), 1):zero():type(torch.type(input))
    self.dissValues = {}
    self.indices = {}
    self.counter = 0
    self.x1SubX3 = {}
    self.x1SubX2 = {}
    self.x2SubX3 = {}
    for i = 1, input:size(1) do
        self.x1SubX3[i] = {}
        self.x1SubX2[i] = {}
        self.x2SubX3[i] = {}

        if not self.dissValues[target[i]] then

            self.indices[target[i]] = torch.find((target - target[i]):ne(0):type(torch.type(torch.FloatTensor())), 1)
            self.dissValues[target[i]] = input:index(1, torch.LongTensor(self.indices[target[i]]))
        end
        local dissValue = self.dissValues[target[i]]

        self.x1SubX3[i][1] = dissValue - input[i]:repeatTensor(dissValue:size(1), 1) --x1SubX3
        self.x1SubX3[i][2] = torch.norm(self.x1SubX3[i][1], 2, 2) --normX1SubX3
        self.x1SubX3[i][3] = torch.exp(self.alpha - self.x1SubX3[i][2]) --expX1SubX3

        for j = 1, input:size(1) do
            if target[i] == target[j] and i ~= j then
                self.x1SubX2[i][j] = {}
                self.x2SubX3[i][j] = {}
                self.x2SubX3[i][j][1] = dissValue - input[j]:repeatTensor(dissValue:size(1), 1) --x2SubX3
                self.x2SubX3[i][j][2] = torch.norm(self.x2SubX3[i][j][1], 2, 2) --normX2SubX3
                self.x2SubX3[i][j][3] = torch.exp(self.alpha - self.x2SubX3[i][j][2]) --expX2SubX3

                if i > j then
                    self.x1SubX2[i][j][1] = -self.x1SubX2[j][i][1]
                    self.x1SubX2[i][j][2] = self.x1SubX2[j][i][2]
                    self.x1SubX2[i][j][3] = self.x1SubX2[j][i][3]
                else
                    self.x1SubX2[i][j][1] = (input[i] - input[j]) --x1SubX2
                    self.x1SubX2[i][j][2] = torch.norm(self.x1SubX2[i][j][1]) --normX1SubX2
                    self.x1SubX2[i][j][3] = torch.exp(self.alpha - self.x1SubX2[i][j][2]) --expX1SubX2
                end
                self.counter = self.x2SubX3[i][j][3]:size(1) * 2
                self.Li[i] = self.x1SubX2[i][j][2] + torch.log(torch.sum(self.x1SubX3[i][3]) + torch.sum(self.x2SubX3[i][j][3]))
            end
        end
    end

    self.output = torch.pow(self.Li, 2):sum() / (2 * self.counter)
    return self.output
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput(input, target)
    self.gradInput = torch.Tensor(input:size()):zero():type(torch.type(input))
    --aa = self:updateGradInput1(input, target)
    self.gradK = {}
    for i = 1, input:size(1) do

        local x1SubX3 = self.x1SubX3[i][1]
        local normX1SubX3 = self.x1SubX3[i][2]

        for j = i + 1, input:size(1) do
            if target[i] == target[j] and i ~= j then
                local subIJ = self.x1SubX2[i][j][1]
                local normSubIJ = self.x1SubX2[i][j][2]
                local diffX1X2 = (subIJ / normSubIJ)

                if not self.gradK[target[i]] then
                    self.gradK[target[i]] = {}
                    local dividedIK = -self.x1SubX3[i][3]
                    local diffX1X3 = torch.cdiv(x1SubX3, normX1SubX3:repeatTensor(1, x1SubX3:size(2)))
                    local x2SubX3 = self.x2SubX3[i][j][1]
                    local normX2SubX3 = self.x2SubX3[i][j][2]
                    local diffX2X3 = torch.cdiv(x2SubX3, normX2SubX3:repeatTensor(1, x2SubX3:size(2)))
                    local dividedJK = -self.x2SubX3[i][j][3]
                    local dividing = torch.exp(self.Li[i] - normSubIJ)[1]

                    local firstDivIK = dividedIK / dividing
                    local firstDivIJ = dividedJK / dividing
                    local diffIK = torch.cmul(diffX1X3, firstDivIK:repeatTensor(1, x1SubX3:size(2)))
                    local diffJK = torch.cmul(diffX2X3, firstDivIJ:repeatTensor(1, x2SubX3:size(2)))
                    self.gradK[target[i]][1] = diffIK
                    self.gradK[target[i]][2] = diffJK
                end

                local gradInput = self.gradInput:index(1, torch.LongTensor(self.indices[target[i]])):add(self.gradK[target[i]][1] + self.gradK[target[i]][2])
                self.gradInput:indexCopy(1, torch.LongTensor(self.indices[target[i]]), gradInput)

                self.gradInput[i] = self.gradInput[i] + diffX1X2 + -self.gradK[target[i]][1]:sum(1)

                self.gradInput[j] = self.gradInput[j] - diffX1X2 + -self.gradK[target[i]][2]:sum(1)
            end
        end
    end
    self.gradInput = torch.cmul(self.Li:expandAs(input), self.gradInput) * 2 / self.counter
    --print(aa)
    --return aa
    return self.gradInput
end


function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput1(input, target)
    self.gradInput1 = torch.Tensor(input:size()):zero():type(torch.type(input))

    for i = 1, input:size(1) do
        for j = 1, input:size(1) do
            if target[i] == target[j] and i ~= j then
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
