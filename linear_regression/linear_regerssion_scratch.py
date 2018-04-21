from mxnet import ndarray as nd
from mxnet import autograd as ag
import matplotlib.pyplot as plt
import random

# ############### 创建数据集 ###############
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

# plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
# plt.show()

# ############### 数据读取 ###############
batch_size = 10


# 每次返回 batch_size 个随机的样本和对应的目标
def data_iter():
	idx = list(range(num_examples))
	random.shuffle(idx)
	for i in range(0, num_examples, batch_size):
		j = nd.array(idx[i: min(i + batch_size, num_examples)])
		yield nd.take(X, j), nd.take(y, j)


# for data, label in data_iter():
# 	print(data, label)
# 	break

# ############### 初始化模型参数 ###############
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
params = [w, b]
# print(params)

# 之后训练需要对这些参数求导来更新它们的值，使损失尽量减小,
# 因此创建它们的梯度.
for param in params:
	param.attach_grad()


# ############### 定义模型 ###############
def net(X):
	return nd.dot(X, w) + b


# ############### 损失函数 ###############
def square_loss(yhat, y):
	return (yhat - y.reshape(yhat.shape)) ** 2


# ############### 优化 ###############
# 通过随机梯度下降来求解.
# 每一步，将模型参数沿着梯度的反方向走特定距离 (learning_rate)
def SGD(params, lr):
	for param in params:
		param[:] = param - lr * param.grad


# ############### 训练 ###############
#  模型函数
def read_fn(X):
	return true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b


epochs = 10
learning_rate = .001

for e in range(epochs):
	total_loss = 0

	for data, label in data_iter():
		with ag.record():
			output = net(data)
			loss = square_loss(output, label)
		loss.backward()
		SGD(params, learning_rate)
		total_loss += nd.sum(loss).asscalar()
	print('Epoch %s, average loss: %f' % (e, total_loss / num_examples))

print(true_w, w)
print(true_b, b)
