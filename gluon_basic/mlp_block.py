from mxnet.gluon import nn
from mxnet import ndarray as nd


# ############### 使用 nn.Block 构造模型 ###############

# nn.Block 类主要提供模型参数的存储、模型计算的定义和自动求导.
# nn.Block 的子类中并没有定义如何求导，或者是 backward 函数.
# MXNet 会使用 autograd 对 forward 自动生成相应的 backward 函数.

# 任意一个 nn.Block 的子类至少实现以下两个函数.
# __init__: 创建模型的参数.
# forward: 定义模型的计算.
class MLP(nn.Block):
	def __init__(self, **kwargs):
		super(MLP, self).__init__(**kwargs)
		with self.name_scope():
			self.hidden = nn.Dense(256, activation='relu')
			self.output = nn.Dense(10)

	def forward(self, x):
		return self.output(self.hidden(x))


net = MLP()
net.initialize()

x = nd.random.uniform(shape=(2, 20))
print(net(x))
print('hidden layer name with default prefix:', net.hidden.name)
print('output layer name with default prefix:', net.output.name)
