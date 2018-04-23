from mxnet.gluon import nn
from mxnet import ndarray as nd


# 不含 with.name_scope().
# 此时，隐藏层和输出层的名字前都不再含指定的前缀 prefix.
class MLP_NO_NAMESCOPE(nn.Block):
	def __init__(self, **kwargs):
		super(MLP_NO_NAMESCOPE, self).__init__(**kwargs)
		self.hidden = nn.Dense(256, activation='relu')
		self.output = nn.Dense(10)

	def forward(self, x):
		return self.output(self.hidden(x))


x = nd.random.uniform(shape=(2, 20))

net = MLP_NO_NAMESCOPE()
net.initialize()

print(net(x))
print('hidden layer name without prefix:', net.hidden.name)
print('output layer name without prefix:', net.output.name)
