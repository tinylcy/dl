from mxnet import gluon
from mxnet import ndarray as nd


def transform(data, label):
	return data.astype('float32') / 255, label.astype('float32')


def load_data_fashion_mnist(batch_size):
	mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
	mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
	train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
	test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
	return train_data, test_data


def accuracy(output, label):
	return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(data_iterator, net):
	acc = 0
	for data, label in data_iterator:
		output = net(data)
		acc += accuracy(output, label)
	return acc / len(data_iterator)
