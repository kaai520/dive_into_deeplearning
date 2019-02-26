from mxnet import nd,autograd
import numpy as np
import matplotlib.pyplot as plt
import random
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon
#求梯度的例子
# x=nd.arange(4).reshape((-1,1))
# x.attach_grad()
# with autograd.record():
#     y = 2 * nd.dot(x.T, x)
# y.backward()
# print(x.grad)

#线性规划原始写法
# n_samples=1000
# n_inputs=2
# X=nd.random_normal(shape=(n_samples,n_inputs))
# true_W=nd.array([2,-3.4]).reshape((-1,1))
# true_b=4.2
# y=nd.dot(X,true_W)+true_b
# y+=1e-2*nd.random_normal(scale=1,shape=y.shape)
# 
# #小批量样本生成函数
# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices) # 样本的读取顺序是随机的
#     for i in range(0, num_examples, batch_size):
#         j = nd.array(indices[i: min(i + batch_size, num_examples)])
#     yield features.take(j), labels.take(j) # take函数根据索引返回对应元素
# 
# w=nd.random_normal(shape=(n_inputs,1))
# b=nd.zeros(shape=1)
# w.attach_grad()
# b.attach_grad()
# def linreg(x, W, c):
#     return nd.dot(x, W) + c
# 
# def squared_loss(y_hat,y):
#     return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# 
# def sgd(params, alpha, batch_size):
#     for param in params:
#         param[:] = param - alpha * param.grad / batch_size
# 
# lr = 0.03
# num_epochs = 200
# net = linreg
# loss = squared_loss
# b_size=10
# for epoch in range(num_epochs):
#     for train_x,train_y in data_iter(b_size,X,y):
#         with autograd.record():
#             l=loss(net(train_x,w,b),train_y)
#         l.backward()
#         sgd([w,b],lr,b_size)
#     print(loss(net(X,w,b),y).mean().asnumpy())
# print(w)
# print(b)

#线性回归简洁实现
num_inputs=2
num_examples=1000
true_w=nd.array([2,-3.4]).reshape((-1,1))
true_b=4.2
features=nd.random.normal(scale=1,shape=(num_examples,num_inputs))
labels=nd.dot(features,true_w)+true_b
labels+=nd.random.normal(scale=0.01,shape=labels.shape)

batch_size=10
dataset = gdata.ArrayDataset(features,labels)
data_iter=gdata.DataLoader(dataset,batch_size,shuffle=True)

#Sequential是串联各层的容器
net=nn.Sequential()
#Dense是全连接层,1表示该层输出的变量的数目，输入变量的数目会自动确定
net.add(nn.Dense(1))
#权重参数高斯随机初始化
net.initialize(init.Normal(sigma=0.01))

#平方损失
loss=gloss.L2Loss()

#定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
#训练
num_epochs=3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        with autograd.record():
            l = loss(net(X),y)
        l.backward()
        trainer.step(batch_size)
    l=loss(net(features),labels)
    print('epoch %d, loss:%f'%(epoch,l.mean().asnumpy()))

print(net[0].weight.data())
print(net[0].bias.data()) 
