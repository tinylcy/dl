from mxnet import ndarray as nd
from mxnet.gluon import nn

# ############### 多层感知机的计算 ###############

# 在 nn.Sequential 里依次添加两个全连接层构造出多层感知机.
# 其中第一层的输出大小为 256，即隐藏层单元个数；第二层的输
# 出大小为 10，即输出层单元个数.
net = nn.Sequential()
with net.name_scope():
	net.add(nn.Dense(256, activation='relu'))
	net.add(nn.Dense(10))

net.initialize()

x = nd.random.uniform(shape=(2, 20))

# 让模型根据输入数据做一次计算
print(net(x))
print('hidden layer: ', net[0])
print('output layer: ', net[1])
