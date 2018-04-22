from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon

# ############### 创建数据集 ###############
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

# ############### 数据读取 ###############
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

# for data, label in data_iter:
# 	print(data, label)
# 	break

# ############### 定义模型 ###############
net = gluon.nn.Sequential()
# 加入一个 Dense 层，它唯一必须定义的参数就是输出节点的个数，在线性模型里面是 1.
net.add(gluon.nn.Dense(1))

# ############### 初始化模型参数 ###############
net.initialize()

# ############### 损失函数 ###############
square_loss = gluon.loss.L2Loss()

# ############### 优化 ###############
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

# ############### 训练 ###############
epochs = 5

for e in range(epochs):
	total_loss = 0
	for data, label in data_iter:
		with ag.record():
			output = net(data)
			loss = square_loss(output, label)
		loss.backward()
		trainer.step(batch_size)
		total_loss += nd.sum(loss).asscalar()

	print('Epoch %d, average loss: %f' % (e, total_loss / num_examples))

# 比较学到的和真实的模型：先从 net 拿到需要的层，然后访问其权重和位移
dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())
