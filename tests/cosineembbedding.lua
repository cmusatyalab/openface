--
-- Created by IntelliJ IDEA.
-- User: cenk
-- Date: 16/01/2017
-- Time: 23:52
-- To change this template use File | Settings | File Templates.
--

require 'nn'
require 'torch'
require 'dpnn'
require 'optim'
require 'image'
require 'torchx'
require 'optim'
require 'xlua'
torch.setdefaulttensortype("torch.FloatTensor")


local TestCosineEmbeddingCriterion, parent = torch.class('nn.TestCosineEmbeddingCriterion', 'nn.Criterion')

function TestCosineEmbeddingCriterion:__init(margin)
    parent.__init(self)
    margin = margin or 0
    self.margin = margin
    self.gradInput = { torch.Tensor(), torch.Tensor() }
    self.sizeAverage = true
end

function TestCosineEmbeddingCriterion:updateOutput(input, y)

    local input1, input2 = input[1], input[2]

    -- keep backward compatibility
    if type(y) == 'number' then
        self._y = self._y or input1.new(1)
        self._y[1] = y
        y = self._y
    end

    if input1:dim() == 1 then
        input1 = input1:view(1, -1)
        input2 = input2:view(1, -1)
    end

    if not self.buffer then
        self.buffer = input1.new()
        self.w1 = input1.new()
        self.w22 = input1.new()
        self.w = input1.new()
        self.w32 = input1.new()
        self._outputs = input1.new()
        -- comparison operators behave differently from cuda/c implementations
        if input1:type() == 'torch.CudaTensor' then
            self._idx = input1.new()
        else
            self._idx = torch.ByteTensor()
        end
    end
    print(input1, input2)
    self.buffer:cmul(input1, input2)
    print("B",self.buffer,"B")
    print(self.buffer)
    self.w1:sum(self.buffer, 2)

    local epsilon = 1e-12
    self.buffer:cmul(input1, input1)
    self.w22:sum(self.buffer, 2):add(epsilon)
    -- self._outputs is also used as a temporary buffer
    self._outputs:resizeAs(self.w22):fill(1)
    self.w22:cdiv(self._outputs, self.w22)
    self.w:resizeAs(self.w22):copy(self.w22)

    self.buffer:cmul(input2, input2)
    self.w32:sum(self.buffer, 2):add(epsilon)
    self.w32:cdiv(self._outputs, self.w32)
    self.w:cmul(self.w32)
    self.w:sqrt()

    self._outputs:cmul(self.w1, self.w)
    self._outputs = self._outputs:select(2, 1)

    y.eq(self._idx, y, -1)
    self._outputs[self._idx] = self._outputs[self._idx]:add(-self.margin):cmax(0)
    y.eq(self._idx, y, 1)
    self._outputs[self._idx] = self._outputs[self._idx]:mul(-1):add(1)

    self.output = self._outputs:sum()

    if self.sizeAverage then
        self.output = self.output / y:size(1)
    end

    return self.output
end

function TestCosineEmbeddingCriterion:updateGradInput(input, y)

    local v1 = input[1]
    local v2 = input[2]
    local not_batch = false

    -- keep backward compatibility
    if type(y) == 'number' then
        self._y = self._y or input1.new(1)
        self._y[1] = y
        y = self._y
    end

    if v1:dim() == 1 then
        v1 = v1:view(1, -1)
        v2 = v2:view(1, -1)
        not_batch = true
    end

    local gw1 = self.gradInput[1]
    local gw2 = self.gradInput[2]
    gw1:resizeAs(v1):copy(v2)
    gw2:resizeAs(v1):copy(v1)

    self.buffer:cmul(self.w1, self.w22)
    gw1:addcmul(-1, self.buffer:expandAs(v1), v1)
    gw1:cmul(self.w:expandAs(v1))

    self.buffer:cmul(self.w1, self.w32)
    gw2:addcmul(-1, self.buffer:expandAs(v1), v2)
    gw2:cmul(self.w:expandAs(v1))

    -- self._idx = self._outputs <= 0
    y.le(self._idx, self._outputs, 0)
    self._idx = self._idx:view(-1, 1):expand(gw1:size())
    gw1[self._idx] = 0
    gw2[self._idx] = 0

    y.eq(self._idx, y, 1)
    self._idx = self._idx:view(-1, 1):expand(gw2:size())
    gw1[self._idx] = gw1[self._idx]:mul(-1)
    gw2[self._idx] = gw2[self._idx]:mul(-1)

    if self.sizeAverage then
        gw1:div(y:size(1))
        gw2:div(y:size(1))
    end

    if not_batch then
        self.gradInput[1]:resize(gw1:size(2))
        self.gradInput[2]:resize(gw2:size(2))
    end

    return self.gradInput
end

function TestCosineEmbeddingCriterion:type(type)
    self._idx = nil
    parent.type(self, type)
    -- comparison operators behave differently from cuda/c implementations
    if type == 'torch.CudaTensor' then
        self._idx = torch.CudaTensor()
    else
        self._idx = torch.ByteTensor()
    end
    return self
end



a = torch.rand(2, 8)
b = torch.rand(2, 8)
y = torch.FloatTensor { -1, 1 }
cri = nn.TestCosineEmbeddingCriterion()
embed = cri:forward({ a, b }, y)
cri:backward({ a, b }, y)

