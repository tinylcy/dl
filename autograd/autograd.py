from mxnet import ndarray as nd
from mxnet import autograd as ag

# ############### 为变量附上梯度 ###############
x = nd.array([[1, 2], [3, 4]])

# 进行求导时，需要存储 x 的导数，通过 NDArray 的方法 attach_grad() 申请对应的空间
x.attach_grad()

# 使用 record() 要求 MXNet 记录需要求导的程序
with ag.record():
	y = x * 2
	z = y * x

# 求导
z.backward()

print('x.grad:', x.grad)


# ############### 对控制流求导 ###############
def f(a):
	b = a * 2
	while nd.norm(b).asscalar() < 1000:
		b = b * 2
	if nd.sum(b).asscalar() > 0:
		c = b
	else:
		c = 100 * b
	return c


a = nd.random_normal(shape=(3, 4))
a.attach_grad()
with ag.record():
	c = f(a)
c.backward()
print(a.grad == c / a)
