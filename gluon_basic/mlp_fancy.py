from mxnet.gluon import nn
from mxnet import ndarray as nd


class FancyMLP(nn.Block):
	def __init__(self, **kwargs):
		super(FancyMLP, self).__init__(**kwargs)
		self.rand_weight = nd.random.uniform(shape=(10, 20))
		with self.name_scope():
			self.dense = nn.Dense(10, activation='relu')

	def forward(self, x):
		x = self.dense(x)
		x = nd.relu(nd.dot(x, self.rand_weight) + 1)
		x = self.dense(x)
		return x


x = nd.random.uniform(shape=(2, 20))

net = FancyMLP()
net.initialize()

print(net(x))
