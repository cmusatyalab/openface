require 'nn'
require 'dpnn'

torch.setdefaulttensortype('torch.FloatTensor')
model = torch.load('/root/openface/models/openface/nn4.small2.v1.t7')
model.save('nn4.small2.v1.resaved.t7')

model:evaluate()
ones = torch.ones(1,3,96,96)
x = model:forward(ones)
torch.save('l'..tostring(#model.modules)..'out.t7',x)


for i=#model.modules,2,-1 do
    model:remove(i)
    x = model:forward(ones)
    torch.save('l'..tostring(i-1)..'out.t7',x)
end
