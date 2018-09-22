import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
import time
from datetime import datetime
from DBN import *


def data_process(filename):     #数据归一化
    fr = open(filename, 'r')
    content = fr.readlines()
    datas = []
    for x in content[1:]:
        i=content[1:].index(x)
        x = x.strip().split(',')
        datas.append([float(k) for k in x[1:-1]])
        if x[-1]=='"setosa"': datas[i].append(0)
        elif x[-1]=='"versicolor"': datas[i].append(1)
        else: datas[i].append(2)
    datas = np.array(datas).astype('float32')
    m,n=datas.shape
    for j in range(n-1):
        meanVal=np.mean(datas[:,j])
        stdVal=np.std(datas[:,j])
        datas[:,j]=(datas[:,j]-meanVal)/stdVal
    return datas

def gen_train_test(dat):        #分割训练集和测试集
    m,n=dat.shape
    np.random.shuffle(dat)
    print(dat)
    traindat=torch.from_numpy(dat[:100,:-1]).float()
    trainlabel=torch.from_numpy(dat[:100,-1]).long()
    testdat = torch.from_numpy(dat[100:,:-1]).float()
    testlabel = torch.from_numpy(dat[100:,-1]).long()
    return traindat,trainlabel,testdat,testlabel

def train_batch(traind,trainl,SIZE=10,SHUFFLE=True,WORKER=2):   #分批处理
    trainset=Data.TensorDataset(traind,trainl)
    trainloader=Data.DataLoader(
        dataset=trainset,
        batch_size=SIZE,
        shuffle=SHUFFLE,
        num_workers=WORKER,  )
    return trainloader

EPOCH=200
BATCH_SIZE=10
LR=0.005

dat=data_process('iris.csv')
print(dat[0]);print(dat.shape)
traind,trainl,testdat,testlabel=gen_train_test(dat)
# print(traind);print(trainl)
train_loader=train_batch(traind,trainl,BATCH_SIZE,SHUFFLE=True,WORKER=0)

# net=torch.nn.Sequential(
#     torch.nn.Linear(4,4),
#     # torch.nn.ReLU(),
#     # torch.nn.Linear(10,5),
#     torch.nn.ReLU(),
#     torch.nn.Linear(4, 3),
# )
#
# # torch.save(net,'iris3(2).pkl')
# # torch.save(net.state_dict(),'iris3_params(2).pkl')
#
# optimizer=torch.optim.SGD(net.parameters(),lr=LR,momentum=0.8)
# loss_func=torch.nn.CrossEntropyLoss()
#
# net.train()
# for epoch in range(EPOCH):
#     for step,(x,y) in enumerate(train_loader):
#         # print(x.data.numpy(),y.data.numpy())
#         # start_time = time.time()
#         b_x=Variable(x)
#         b_y=Variable(y)
#
#         output=net(b_x)
#         prediction = torch.max(output, 1)[1]
#         # print(prediction);print(output);print(b_y)
#
#         loss=loss_func(output,b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if step % 5==0:
#             print('Epoch: ', epoch, 'step:',step,'| train loss: %.4f' % loss.data.numpy(), )
#
# net.eval()
# test_x=Variable(testdat);test_y=Variable(testlabel)
# test_out=net(test_x)
# test_pred=torch.max(test_out,1)[1]
# pre_val = test_pred.data.squeeze().numpy()
# y_val = test_y.data.squeeze().numpy()
# print(pre_val);print(y_val)
# accuracy = float((pre_val == y_val).astype(int).sum()) / float(test_y.size(0))
# print('| test accuracy: %.2f' % accuracy)

dat=data_process('iris.csv')
traind,trainl,testd,testl=gen_train_test(dat)
# print(traind);print(trainl)
train_loader=train_batch(traind,trainl,20,SHUFFLE=True,WORKER=2)
# print(len(train_loader),len(test_loader))

a,b=train_and_test(traind,trainl,testd,testl,train_loader)