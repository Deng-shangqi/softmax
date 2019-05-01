import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle


# mnist = input_data.read_data_sets('data', one_hot=True)
# x_batch, y_batch = mnist.train.next_batch(10)
# for i in range(10):
#     plt.imshow(x_batch[i].reshape(28, 28), cmap='Greys')
# plt.show()


def softmax(x):

    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom  = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)  # 每一行减去它的最大值取指数
        denominator = np.apply_along_axis(denom, 1, x)  # 每一行的数相加的倒数

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))
        x = x * denominator  # 归一化

    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    # print(x)

    return x

if __name__ == '__main__':
    start = time.time()

    mnist = input_data.read_data_sets('data', one_hot=True)
    # x_batch, y_batch = mnist.train.next_batch(1000)

    # dim1 = x_batch.shape[1]  # 每个x有多少个输入
    # dim2 = y_batch.shape[1]  # 输出层有多少个输出

    dim1 = 784  # 每个x有多少个输入
    dim2 = 10  # 输出层有多少个输出

    w = np.zeros((dim1, dim2))
    for i in range(10000):
        x_batch, y_batch = mnist.train.next_batch(100)
        a = softmax(np.dot(x_batch, w))  # 预测值
        error = a - y_batch
        w = w - 0.01 * np.dot(x_batch.transpose(),  error)
    # x, y = mnist.train.next_batch(1)
    # for i in range(20):
    #     x = x_batch[i]
    #     y = y_batch[i]
    #     p = softmax(np.dot(x, w))
    #     p = list(p)
    #     y = list(y)
    #     print("the corect number is ->", y.index(max(y)), "pridict", p.index(max(p)))

    # x1, y1 = mnist.test.next_batch(1000)
    x1 = mnist.test.images
    y1 = mnist.test.labels
    count = 0
    for i in range(mnist.test.num_examples):
        x = x1[i]
        y = y1[i]
        p = softmax(np.dot(x, w))
        p = list(p)
        y = list(y)
        # print("the corect number is ->", y.index(max(y)), "pridict", y.index(max(y)))
        if y.index(max(y)) == p.index(max(p)):
            count += 1
    print("accuracy:", count / mnist.test.num_examples)
    end = time.time()
    print("time:", end-start)


    # softmax(np.array([3, 1, -3]))

