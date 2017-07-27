local MultiModalMarginRankingCriterion, parent = torch.class('nn.MultiModalMarginRankingCriterion', 'nn.Criterion')

-- loss(x, y) = max(0, -y * (x[1] - x[2]) + margin) +
--              max(0, -y * (x[1] - x[3]) + margin)

function MultiModalMarginRankingCriterion:__init(margin)
    parent.__init(self)
    margin = margin or 1
    self.margin = margin
    self.gradInput = { torch.Tensor(1), torch.Tensor(1), torch.Tensor(1) }
    self.sizeAverage = true
end

function MultiModalMarginRankingCriterion:updateOutput(input, y)
    if torch.type(y) == 'number' then -- non-batch mode
        --self.output = math.max(0, -y * (input[1][1] - input[2][1]) + self.margin)
        self.output = math.max(0, -y * (input[1][1] - input[2][1]) + self.margin)
                + math.max(0, -y * (input[1][1] - input[3][1]) + self.margin)
    else

        self._output1 = self._output1 or input[1]:clone()
        self._output1:resizeAs(input[1])
        self._output1:copy(input[1])
        self._output2 = self._output1:clone()

        -- max(0, -y * (x[1] - x[2]) + margin) +
        self._output1:add(-1, input[2])
        self._output1:mul(-1):cmul(y)
        self._output1:add(self.margin)

        self._output1:cmax(0)

        -- max(0, -y * (x[1] - x[3]) + margin)
        self._output2:add(-1, input[3])
        self._output2:mul(-1):cmul(y)
        self._output2:add(self.margin)

        self._output2:cmax(0)


        self.output = self._output1:sum() + self._output2:sum()

        if self.sizeAverage then
            self.output = self.output / y:size(1)
        end
    end

    return self.output
end




-- loss(x, y) = max(0, -y * (x[1] - x[2]) + margin) +
--              max(0, -y * (x[1] - x[3]) + margin)

function MultiModalMarginRankingCriterion:updateGradInput(input, y)
    if torch.type(y) == 'number' then -- non-batch mode
        --local dist = -y * (input[1][1] - input[2][1]) + self.margin
        local dist1 = -y * (input[1][1] - input[2][1]) + self.margin
        local dist2 = -y * (input[1][1] - input[3][1]) + self.margin

        if dist1 < 0 then
            self.gradInput[1][1] = 0;
            self.gradInput[2][1] = 0;
        else
            self.gradInput[1][1] = -y
            self.gradInput[2][1] = y
        end

        if dist2 < 0 then
            self.gradInput[1][1] = 0 + self.gradInput[1][1];
            self.gradInput[3][1] = 0;
        else
            self.gradInput[1][1] = -y + self.gradInput[1][1];
            self.gradInput[3][1] = y
        end

    else
        self.dist1 = self.dist1 or input[1].new()
        self.dist1 = self.dist1:resizeAs(input[1]):copy(input[1])
        local dist1 = self.dist1
        self.dist2 = self.dist1:clone()
        local dist2 = self.dist2

        dist1:add(-1, input[2])
        dist1:mul(-1):cmul(y)
        dist1:add(self.margin)

        dist2:add(-1, input[3])
        dist2:mul(-1):cmul(y)
        dist2:add(self.margin)

        self.mask1 = self.mask1 or input[1].new()
        self.mask1 = self.mask1:resizeAs(input[1]):copy(dist1)
        local mask1 = self.mask1

        self.mask2 = self.mask2 or input[1].new()
        self.mask2 = self.mask2:resizeAs(input[1]):copy(dist2)
        local mask2 = self.mask2

        mask1:ge(dist1, 0)
        mask2:ge(dist2, 0)

        self.gradInput[1]:resize(dist1:size())
        self.gradInput[2]:resize(dist1:size())
        self.gradInput[3]:resize(dist2:size())

        self.gradInput[1]:copy(mask1)
        self.gradInput[1]:mul(-1):cmul(y)
        self.gradInput[2]:copy(mask1)
        self.gradInput[2]:cmul(y)

        local gradInput_ = self.gradInput[1]:clone()
        gradInput_:copy(mask2)
        gradInput_:mul(-1):cmul(y)
        self.gradInput[1]:add(gradInput_)
        self.gradInput[3]:copy(mask2)
        self.gradInput[3]:cmul(y)


        if self.sizeAverage then
            self.gradInput[1]:div(y:size(1))
            self.gradInput[2]:div(y:size(1))
            self.gradInput[3]:div(y:size(1))
        end
    end
    return self.gradInput
end
