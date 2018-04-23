from mxnet.gluon import nn
from mxnet import ndarray as nd


class NestMLP(nn.Block):
	def __init__(self, **kwargs):
		super(NestMLP, self).__init__(**kwargs)
		self.net = nn.Sequential()
		with self.name_scope():
			self.net.add(nn.Dense(64, activation='relu'))
			self.net.add(nn.Dense(32, activation='relu'))
			self.dense = nn.Dense(16, activation='relu')

	def forward(self, x):
		return self.dense(self.net(x))


net = nn.Sequential()
net.add(NestMLP())
net.add(nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
print(net(x))
