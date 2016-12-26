--
-- Created by IntelliJ IDEA.
-- User: bau_fbe
-- Date: 26.12.2016
-- Time: 13:31
-- To change this template use File | Settings | File Templates.
--

local BatchKLDivCriterion, parent = torch.class('nn.BatchKLDivCriterion', 'nn.Criterion')
-- This function is only for handling the output of softmax layer

local epsilon = 1e-35 --to avoid calculating log(0) or dividing 0

function BatchKLDivCriterion:__init(margin)
   parent.__init(self)
   self.sizeAverage = false
   self.margin = margin or 2 -- 2 is the empirical value for mnist
end

function BatchKLDivCriterion:KLDiv(target, input)
   -- The function calculates the KL-divergence
   -- target and input must be dim==1
   self.buf_t = self.buf_t or target.new()
   self.buf_t:resizeAs(target)
   self.buf_i = self.buf_i or input.new()
   self.buf_i:resizeAs(input)

   -- KL(T||I) = \sum T(logT-logI)
   self.buf_t = target + epsilon
   self.buf_i = input  + epsilon
   self.buf_t:log() -- logT
   self.buf_i:log() -- logI
   self.buf_t = self.buf_t - self.buf_i --logT-logI
   local dist = self.buf_t:cmul(target):sum() -- a number

   return dist
end

function BatchKLDivCriterion:is_pair_labeled(i,j,target)
  if target:dim()==2 then
    return target[i][j]~=0
  end
  return target[i]>0 and target[j]>0

end

function BatchKLDivCriterion:is_simi_pair(i,j,target)
  if target:dim()==2 then
    return target[i][j]==1
  end
  return target[i]==target[j]
end

function BatchKLDivCriterion:loss(p, q, input, simi, cache_idx, load_cache)
   cache_idx = cache_idx or 0
   load_cache = load_cache or false
   if load_cache then return self.loss_cache[cache_idx] end --read cache to speedup

   local loss = self:KLDiv(input[p], input[q])
   if not simi then --hinge lost
      loss = self.margin - loss
      if loss<0 then loss=0 end
   end
   self.loss_cache[cache_idx] = loss  --save the loss to cache
   return loss
end

function BatchKLDivCriterion:updateOutput(input, target)
   -- Unlabeled data is handled: target[i]=0 means no label for sample i
   -- assert(input:dim() == 2)
   -- assert(input:size(1) > 1)
   print(input,target)
   self.output = 0
   local pair_count = 0
   local pair_idx = 0

   -- prepare loss_cache buffer for saving some computation
   if target:dim()==1 then
      pair_count = target:size(1)*(target:size(1)+1)/2 - target:size(1)
   elseif target:size(1)==target:size(2) then
      pair_count = target:ne(0):sum()/2
   else
      pair_count = target:size(1)
   end
   self.loss_cache = self.loss_cache or input[1].new()
   self.loss_cache:resize(pair_count*2):fill(0) -- for saving the loss of both direction

   -- We handle 3 types of 'target':
   -- 1. nx1 class label - so we could easily replace CrossEntropyCriterion by BatchKLDivCriterion
   -- 2. nxn relationship matrix
   -- 3. nx3 tuple

   -- target is nx1 class label or nxn relationship matrix
   if target:dim()==1 or target:size(1)==target:size(2) then
      for i=1,input:size(1) do
         for j=1,input:size(1) do
            if i~=j and self:is_pair_labeled(i,j,target) then
               pair_idx = pair_idx+1
               self.output = self.output + self:loss(j, i, input, self:is_simi_pair(i,j,target), pair_idx)
            end
         end
      end
   -- target is nx3: (i,j,relationship); relationship= 1:similar pair, 0: dissimilar pair
   else
      local test_cost = 0
      for p=1,pair_count do
         local i,j,s = target[p][1],target[p][2],target[p][3]
         s = s==1
         self.output = self.output + self:loss(j, i, input, s, p) + self:loss(i, j, input, s, pair_count+p)
      end
   end

   if pair_count>0 then
      self.output = self.output / pair_count
   end

   return self.output
end

function BatchKLDivCriterion:updateGradInput(input, target)
   --assert(input:dim() == 2)
   --assert(input:size(1) > 1)
   local pair_count = 0
   local pair_idx = 0
   local factor = 0 -- Indicator of similar pair (similar: 1; dissimilar: -1)
   self.gradInput = self.gradInput or input.new()
   if self.gradInput:type()~=input:type() then self.gradInput=input.new() end
   self.gradInput:resizeAs(input):fill(0)
   self.buf = self.buf or input[1].new()
   self.buf:resizeAs(input[1])

   if target:dim()==1 then
      pair_count = target:size(1)*(target:size(1)+1)/2 - target:size(1)
   elseif target:size(1)==target:size(2) then
      pair_count = target:ne(0):sum()/2
   else
      pair_count = target:size(1)
   end

   -- target is nx1 class label or nxn relationship matrix
   if target:dim()==1 or target:size(1)==target:size(2) then
      for i=1,input:size(1) do
         for j=1,input:size(1) do
            if i~=j and self:is_pair_labeled(i,j,target) then
               pair_idx = pair_idx+1
               if self:is_simi_pair(i,j,target) then factor=1 else factor=-1 end
               local loss = self:loss(j, i, input, factor==1, pair_idx, true)
               if loss~=0 then
                  self.buf:copy(input[j]):cdiv(input[i]+epsilon):mul(-1)
                  self.gradInput[i] = self.gradInput[i] + self.buf:mul(factor)
               end
            end
         end
      end
   -- target is nx3: (i,j,relationship); relationship= 1:similar pair, 0: dissimilar pair
   else
      for p=1,pair_count do
         local i,j,s = target[p][1],target[p][2],target[p][3]
         s = s==1
         local loss_i = self:loss(j, i, input, s, p, true) --read the loss from cache
         local loss_j = self:loss(i, j, input, s, pair_count+p, true)
         if s then factor=1 else factor=-1 end
         -- Implementation trick: Ignore the gradient when loss=0 (even it is a similar pair).
         -- The trick saves some computation and doesn't affect the result.
         -- Do the clustering multiple times with random initialization is the key to reach lower training error.
         -- dloss(p,q)/dq_i = -p_i/q_i
         if loss_i~=0 then
            self.buf:copy(input[j]):cdiv(input[i]+epsilon):mul(-1)
            self.gradInput[i] = self.gradInput[i] + self.buf:mul(factor)
         end
         -- dloss(q,p)/dp_i = -q_i/p_i
         if loss_j~=0 then
            self.buf:copy(input[i]):cdiv(input[j]+epsilon):mul(-1)
            self.gradInput[j] = self.gradInput[j] + self.buf:mul(factor)
         end

      end
   end

   -- average by size
   if self.sizeAverage then self.gradInput = self.gradInput:div(pair_count) end

   return self.gradInput
end
