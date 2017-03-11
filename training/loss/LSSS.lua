local LiftedStructuredSimilaritySoftmaxCriterion, parent = torch.class('nn.LiftedStructuredSimilaritySoftmaxCriterion', 'nn.Criterion')

function LiftedStructuredSimilaritySoftmaxCriterion:__init(alpha)
    parent.__init(self)
    self.alpha = alpha or 1
    self.Li = torch.Tensor()
    self.gradInput = {}
    self.counter = 0
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateOutput(input, mapper)
    local a = input[1]
    local p = input[2]
    local n = input[3]
    self.Dij = (a - p):norm(2, 2)
    self.Dik = torch.exp(self.alpha - (a - n):norm(2, 2))
    self.Djl = torch.exp(self.alpha - (p - n):norm(2, 2))


    self.Li = torch.Tensor(a:size(1), 1):type(a:type()):zero()
    self.li = {}
    local items = {}
    for i = 1, table.getn(mapper) do
        local totalDik = 0
        local totalDjl = 0
        local counter = 0
        self.li[i] = {}
        items[mapper[i][1]] = 0
        for j = 1, table.getn(mapper) do
            if mapper[i][1] == mapper[j][1] and mapper[i][2] == mapper[j][2] then
                totalDik = totalDik + self.Dik[j]
                totalDjl = totalDjl + self.Djl[j]
                counter = counter + 1
                self.li[i][j] = {}
                self.li[i][j]["dik"] = self.Dik[j]
                self.li[i][j]["djl"] = self.Djl[j]
            end
        end
        self.li[i]["dij"] = self.Dij[i]
        self.Li[i] = (self.Dij[i] + torch.log(totalDik + totalDjl)) / (2 * counter)
    end
    self.counter = table.getn(items)
    self.output = self.Li:sum()
    return self.output
end

function LiftedStructuredSimilaritySoftmaxCriterion:updateGradInput(input, mapper)
    local a = input[1]
    local p = input[2]
    local n = input[3]

    self.gradInput = torch.Tensor(self.counter, a:size(2)):type(a:type()):zero()
    for i = 1, table.getn(mapper) do

        local tar = torch.cmul(self.Li[i]:gt(0):repeatTensor(a[i]:size(), 1):type(a:type()), self.Dij[i]:repeatTensor(a[i]:size(), 1):type(a:type()))
        for j = 1, table.getn(mapper) do
            if mapper[i][1] == mapper[j][1] and mapper[i][2] == mapper[j][2] then
                local tar1 = torch.cmul(self.Li[i]:gt(0):repeatTensor(a[i]:size(), 1):type(a:type()), torch.cmul(torch.cdiv(-self.Dik[j], torch.exp(self.Li[i] - self.li[i]["dij"])), self.li[i]["dij"]):repeatTensor(a[i]:size(), 1):type(a:type()))
                self.gradInput[mapper[j][3]] = self.gradInput[mapper[j][3]] + (a[i] - n[i]):cmul(tar1)
            end
        end
        self.gradInput[mapper[i][1]] = self.gradInput[mapper[i][1]] + (a[i] - p[i]):cmul(tar)
        self.gradInput[mapper[i][2]] = self.gradInput[mapper[i][2]] + (p[i] - a[i]):cmul(tar)
    end
    self.output = self.Li:sum()



    return self.gradInput
end
