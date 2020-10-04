import torch
import torch.nn as nn 

class OurModule(nn.Module):
	def __init__(self, num_inputs, num_classes, drop_prob=0.3):
		super(OurModule, self).__init__()
		self.pipeline = nn.Sequential(
			nn.Linear(num_inputs, 5),
			nn.ReLU(),
			nn.Linear(5,20),
			nn.ReLU(),
			nn.Linear(20,num_classes),
			nn.Dropout(p=drop_prob),
			nn.Softmax(dim=1)
			)
	def forward(self, x):
		return self.pipeline(x)

net = OurModule(num_inputs=2, num_classes=3)
v = torch.FloatTensor([[2,3]])
out = net(v)
print(net)
print(out)
l = nn.Linear(2,5)
v = torch.FloatTensor([1,2])
print(v,l(v))
s = nn.Sequential(
	nn.Linear(2,5),
	nn.ReLU(),
	nn.Linear(5,20),
	nn.ReLU(),
	nn.Linear(20,10),
	nn.Dropout(p=0.3),
	nn.Softmax(dim=1))
print(s)
print(s(torch.FloatTensor([[1,2]])))