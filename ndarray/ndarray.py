from mxnet import ndarray as nd
import numpy as np

x = nd.zeros((3, 4))
print(x)

x = nd.ones((3, 4))
print(x)

x = nd.array([[1, 2], [3, 4]])
print(x)

# 创建随机数组，每个元素的值都是随机采样而来，经常被用于初始化模型参数
y = nd.random_normal(0, 1, shape=(3, 4))
print(y)
print(y.shape)
print(y.size)

x = nd.random_normal(0, 1, shape=(3, 4))
print(x)
print(x + y)
print(x * y)
# 指数运算.
print(nd.exp(y))
# 转置
print(nd.dot(x, y.T))

# 广播
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
print('a:', a)
print('b:', b)
print('a+b:', a + b)

# 跟 Numpy 的转换
x = np.ones((2, 3))
y = nd.array(x)
z = y.asnumpy()
print([z, y])

# 替换操作
x = nd.ones((3, 4))
y = nd.ones((3, 4))
before = id(y)
y = y + x
print(id(y) == before)

z = nd.zeros_like(x)
before = id(z)
z[:] = x + y
print(id(z) == before)

nd.elemwise_add(x, y, out=z)
print(id(z) == before)

# 截取 (Slicing)
x = nd.arange(0, 9).reshape((3, 3))
print('x:', x)
print(x[1:3])

x[1, 2] = 9.0
print(x)

# 多维截取
x = nd.arange(0, 9).reshape((3, 3))
print(x[1:3, 1:3])
# 多维写入
x[1:3, 1:3] = 9.0
print(x)
