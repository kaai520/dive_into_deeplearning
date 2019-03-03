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
import zipfile
import math
import time
#这个随机分布命名太长了
#矩阵连接
# normal=nd.random.normal
# X,W_xh=normal(shape=(3,1)),normal(shape=(1,4))
# H,W_hh=normal(shape=(3,4)),normal(shape=(4,4))
# h1=nd.dot(X,W_xh)+nd.dot(H,W_hh)
# print(h1)
# print(nd.concat(X,H,dim=1))
# print(nd.concat(W_xh,W_hh,dim=0))
# h2=nd.dot(nd.concat(X,H,dim=1),nd.concat(W_xh,W_hh,dim=0))
# print(h2)

#读取数据集
# with zipfile.ZipFile('jaychou_lyrics.txt.zip') as zin:
#     with zin.open('jaychou_lyrics.txt') as f:
#         corpus_chars=f.read().decode('utf-8')
#
# corpus_chars=corpus_chars.replace('\n',' ').replace('\r',' ')
# corpus_chars=corpus_chars[:10000]
#
# idx2char=list(set(corpus_chars))
# char2idx=dict(zip(idx2char,range(len(idx2char))))
# corpus_indices = [char2idx[char] for char in corpus_chars]
# print(corpus_indices[:10])
#
# def data_iter_random(sample_indices,batch_size,num_steps,ctx=None):
#     #减1是因为Y要加1
#     num_examples=(len(sample_indices)-1)//num_steps
#     epoch_size=num_examples//batch_size
#     example_indices=list(range(num_examples))
#     random.shuffle(example_indices)
#
#     def _data(pos):
#         return sample_indices[pos:pos+num_steps]
#
#     for i in range(epoch_size):
#         #每次读取batch_size个随机样本
#         i=i*batch_size
#         batch_indices=example_indices[i:i+batch_size]
#         X=[_data(j*num_steps) for j in batch_indices]
#         Y=[_data(j*num_steps+1) for j in batch_indices]
#         yield nd.array(X,ctx),nd.array(Y,ctx)
#
# def data_iter_consecutive(sample_indices,batch_size,num_steps,ctx=None):
#     sample_indices=nd.array(sample_indices,ctx=ctx)
#     data_len=len(sample_indices)
#     batch_len=data_len//batch_size
#     indices=sample_indices[:batch_size*batch_len].reshape((batch_size,batch_len))
#     epoch_size=(batch_len-1)//num_steps
#     print(indices)
#     for i in range(epoch_size):
#         i=i*num_steps
#         X = indices[:, i: i + num_steps]
#         Y = indices[:, i + 1: i + num_steps + 1]
#         yield X, Y
#
# my_seq=list(range(30))
# for X, Y in data_iter_consecutive(my_seq, batch_size=3, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')

with zipfile.ZipFile('jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars=f.read().decode('utf-8')

corpus_chars=corpus_chars.replace('\n',' ').replace('\r',' ')
corpus_chars=corpus_chars[:10000]

idx_to_char=list(set(corpus_chars))
char_to_idx=dict(zip(idx_to_char, range(len(idx_to_char))))
corpus_indices = [char_to_idx[char] for char in corpus_chars]
vocab_size = len(char_to_idx)

def to_onehot(X,size):
    return [nd.one_hot(x,size) for x in X.T]

num_inputs=vocab_size
num_hiddens=256
num_outputs=vocab_size
ctx=d2l.try_gpu()

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01,shape=shape,ctx=ctx)
    #隐藏层参数
    W_xh=_one((num_inputs,num_hiddens))
    W_hh=_one((num_hiddens,num_hiddens))
    b_h=nd.zeros(num_hiddens,ctx=ctx)
    #输出层参数
    W_hq=_one((num_hiddens,num_outputs))
    b_q=nd.zeros(num_outputs,ctx=ctx)
    # 附上梯度
    params=[W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.attach_grad()
    return params

def init_rnn_state(batch_size,num_hiddens,ctx):
    return (nd.zeros(shape=(batch_size,num_hiddens),ctx=ctx),)

def rnn(inputs,state,params):
    #inputs和outputs皆是num_steps个形状为(batch_size,vocab_size)的矩阵
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        H=nd.tanh(nd.dot(X,W_xh)+nd.dot(H,W_hh)+b_h)
        Y=nd.dot(H,W_hq)+b_q
        outputs.append(Y)
    return outputs,(H,)

def predict_rnn(prefix,num_chars,rnn,params,init_rnn_state,
                num_hiddens,vocab_size,ctx,idx_to_char,char_to_idx):
    state=init_rnn_state(1,num_hiddens,ctx)
    output=[char_to_idx[prefix[0]]]
    for t in range(num_chars+len(prefix)-1):
        #将上一时间步的输出作为当前时间步的输入
        X=to_onehot(nd.array([output[-1]],ctx=ctx),vocab_size)
        #计算输出和更新隐藏状态
        (Y,state)=rnn(X,state,params)
        #下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t<len(prefix)-1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])

def grad_clipping(params,theta,ctx):
    norm=nd.array([0],ctx=ctx)
    for param in params:
        norm+=(param.grad**2).sum()
    norm=norm.sqrt().asscalar()
    if norm>theta:
        for param in params:
            param.grad[:] *=theta/norm

def train_and_predict_rnn(rnn,get_params,init_rnn_state,num_hiddens,
                          vocab_size,ctx,corpus_indices,idx_to_char,
                          char_to_idx,is_random_iter,num_epochs,num_steps,
                          lr,clipping_theta,batch_size,pred_period,
                          pred_len,prefixes):
    if is_random_iter:
        data_iter_fn=d2l.data_iter_random
    else:
        data_iter_fn=d2l.data_iter_consecutive
    params=get_params()
    loss=gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        #如果使用相邻采样，在epoch开始是初始化隐藏状态
        if not is_random_iter:
            state=init_rnn_state(batch_size,num_hiddens,ctx)
        l_sum,n,start=0.0,0,time.time()
        data_iter=data_iter_fn(corpus_indices,batch_size,num_steps,ctx)
        for X,Y in data_iter:
            #如使用随机采样，在每个小批量更新前初始化隐藏状态
            if is_random_iter:
                state=init_rnn_state(batch_size,num_hiddens,ctx)
            else: #否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach()
            with autograd.record():
                inputs=to_onehot(X,vocab_size)
                #outputs有num_steps个形状为(batch_size,vocab_size)
                (outputs,state)=rnn(inputs,state,params)
                #拼接之后形状为(num_steps*batch_size,vocab_size)
                outputs=nd.concat(*outputs,dim=0)
                #Y的形状是(batch_size,num_steps),转置后再变成长度为
                #batch*num_steps的向量，这样跟输出的行一一对应
                y=Y.T.reshape((-1,))
                #使用交叉熵损失计算平均分类误差
                l=loss(outputs,y).mean()
            l.backward()
            grad_clipping(params,clipping_theta,ctx)#裁剪梯度
            d2l.sgd(params,lr,1)#梯度不用做平均
            l_sum+=l.asscalar()*y.size
            n+=y.size

        if (epoch+1)%pred_period==0:
            print('epoch %d,perplexity %f,time %.2f sec'%
                  (epoch+1,math.exp(l_sum/n),time.time()-start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

#随机采样
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)

#连续采样
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
