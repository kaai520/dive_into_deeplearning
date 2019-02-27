from mxnet import nd,autograd
import numpy as np
import matplotlib.pyplot as plt
import random
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon
import d2lzh as d2l

#softmax回归从零实现
# batch_size = 256
# #内部已经分好一个一个小批量了
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#
# #numpy softmax
# def np_softmax(X):
#     x_exp=np.exp(X)
#     partition=x_exp.sum(axis=1,keepdims=True)
#     return x_exp/partition
#
# #mxnet softmax
# def softmax(X):
#     x_exp = X.exp()
#     partition = x_exp.sum(axis=1, keepdims=True)
#     return x_exp / partition
#
# num_inputs=784
# num_outputs=10
#
# W=nd.random.normal(scale=0.01,shape=(num_inputs,num_outputs))
# b=nd.zeros(num_outputs)
# W.attach_grad()
# b.attach_grad()
#
# def net(X):
#     return softmax(nd.dot(X.reshape((-1,num_inputs)),W)+b)
#
# def cross_entropy(y_hat,y):
#     return -nd.pick(y_hat,y).log()
#
# def accuracy(y_hat,y):
#     return (y_hat.argmax(axis=1)==y.astype('float')).mean().asscalar()
#
# def evaluate_accuracy(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X, y in data_iter:
#         y = y.astype('float32')
#         acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
#         n += y.size
#     return acc_sum / n
#
# num_epochs,lr=10,0.1
# def train_ch3(net,train_iter,
#               test_iter,loss,
#               num_epochs,batch_size,
#               params=None,lr=None,trainer=None):
#     for epoch in range(num_epochs):
#         train_l_sum,train_acc_sum,n=0.0,0.0,0
#         for X,y in train_iter:
#             with autograd.record():
#                 y_hat=net(X)
#                 l=loss(y_hat,y).sum()
#             l.backward()
#             if trainer is None:
#                 d2l.sgd(params,lr,batch_size)
#             else:
#                 trainer.step(batch_size)
#             y=y.astype('float32')
#             train_l_sum+=l.asscalar()
#             train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
#             n += y.size
#         test_acc=evaluate_accuracy(test_iter,net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
#               % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
#
# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
#           [W, b], lr)
#
# for X, y in test_iter:
#     break
#
# true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
# pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
# titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
#
# d2l.show_fashion_mnist(X[0:9], titles[0:9])
# plt.show()

#softmax回归简洁实现

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
