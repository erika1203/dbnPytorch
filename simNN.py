'''
2018-09-17
新材料企业数据
回归
'''

import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.utils.data as Data
import time


def data_process(filename):     #数据归一化
    fr = open(filename, 'r')
    content = fr.readlines()
    datas = []
    for x in content[1:]:
        k=content[1:].index(x)
        x = x.strip().split(' ')
        datas.append([float(i) for i in x[:-2]])
        if x[-1]=='危险': datas[k].append(0)
        elif x[-1]=='可疑': datas[k].append(1)
        else: datas[k].append(2)
    datas = np.array(datas).astype('float32')
    m,n=datas.shape

    maxMat=np.max(datas,0)
    minMat=np.min(datas,0)
    diffMat=maxMat-minMat
    for j in range(n-1):
        datas[:,j]=(datas[:,j]-minMat[j])/diffMat[j]

    # for j in range(n-1):
    #     meanVal=np.mean(datas[:,j])
    #     stdVal=np.std(datas[:,j])
    #     datas[:,j]=(datas[:,j]-meanVal)/stdVal

    # print(datas[0])
    return datas

def gen_train_test(dat):        #分割训练集和测试集
    np.random.shuffle(dat)
    traindat=torch.from_numpy(dat[:700,:-1]).float()
    trainlabel=torch.from_numpy(dat[:700,-1]).long()
    testdat = torch.from_numpy(dat[700:,:-1]).float()
    testlabel = torch.from_numpy(dat[700:,-1]).long()
    # print(dat[0])
    return traindat,trainlabel,testdat,testlabel

def train_batch(traind,trainl,SIZE=10,SHUFFLE=True,WORKER=2):   #分批处理
    trainset=Data.TensorDataset(traind,trainl)
    trainloader=Data.DataLoader(
        dataset=trainset,
        batch_size=SIZE,
        shuffle=SHUFFLE,
        num_workers=WORKER,  )
    return trainloader

# class Net(torch.nn.Module):
#     def __init__(self,n_feature,n_h1,n_h2,n_h3,n_output):
#         super(Net,self).__init__()
#         self.hidden1=torch.nn.Linear(n_feature,n_h1)
#         self.hidden2=torch.nn.Linear(n_h1,n_h2)
#         self.hidden3 = torch.nn.Linear(n_h2, n_h3)
#         self.predict=torch.nn.Linear(n_h3,n_output)
#
#     def forward(self, x):
#         x=F.relu(self.hidden1(x))
#         x = F.relu(self.hidden2(x))
#         x = F.relu(self.hidden3(x))
#         x=self.predict(x)
#         out = F.softmax(x, dim=1)
#         return out

def train_and_test(epoch,batch_size,lr,momentum):
    EPOCH=epoch
    BATCH_SIZE=batch_size
    LR=lr

    dat=data_process('data.txt')
    traind,trainl,testdat,testlabel=gen_train_test(dat)
    # print(traind);print(trainl)
    train_loader=train_batch(traind,trainl,BATCH_SIZE,SHUFFLE=True,WORKER=2)

    # net=Net(26,30,20,10,3)

    net=torch.nn.Sequential(
        torch.nn.Linear(26,11),
        torch.nn.ReLU(),
        # torch.nn.Linear(30,30),
        # torch.nn.ReLU(),
        # torch.nn.Linear(30,15),
        # torch.nn.ReLU(),
        # torch.nn.Linear(100,80),
        # torch.nn.ReLU(),
        # torch.nn.Linear(80,50),
        # torch.nn.ReLU(),
        # torch.nn.Linear(50,30),
        # torch.nn.ReLU(),
        # torch.nn.Linear(30,20),
        # torch.nn.ReLU(),

        torch.nn.Linear(11,6),
        torch.nn.ReLU(),
        torch.nn.Linear(6,11),
        torch.nn.ReLU(),
        torch.nn.Linear(11, 11),
        torch.nn.ReLU(),
        torch.nn.Linear(11,3),
    )

    # torch.save(net,'net2(2).pkl')
    # torch.save(net.state_dict(),'net2_params(2).pkl')

    optimizer=torch.optim.SGD(net.parameters(),lr=LR,momentum=momentum)
    loss_func=torch.nn.CrossEntropyLoss()

    net.train()

    start_time = time.time()
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_loader):
            # print(x.data.numpy(),y.data.numpy())

            b_x=Variable(x)
            b_y=Variable(y)

            output=net(b_x)
            prediction = torch.max(output, 1)[1]
            # print(prediction);print(output);print(b_y)

            loss=loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if step % 20==0:
                # test_x=Variable(testdat);test_y=Variable(testlabel)
                # test_out=net(test_x)
                # test_pred=torch.max(test_out,1)[1]
                # pre_val = test_pred.data.squeeze().numpy()
                # y_val = test_y.data.squeeze().numpy()
                # print(pre_val);print(y_val)
                # accuracy = float((pre_val == y_val).astype(int).sum()) / float(test_y.size(0))
                # print('Epoch: ', epoch, 'step:',step,'| train loss: %.4f' % loss.data.numpy(), )
                      # '| test accuracy: %.2f' % accuracy)
    duration = time.time() - start_time
    # print('Duration:%.4f' % duration)

    net.eval()
    test_x=Variable(testdat);test_y=Variable(testlabel)
    test_out=net(test_x)
    test_pred=torch.max(test_out,1)[1]
    pre_val = test_pred.data.squeeze().numpy()
    y_val = test_y.data.squeeze().numpy()
    # print('prediciton:',pre_val);print('true value:',y_val)
    accuracy = float((pre_val == y_val).astype(int).sum()) / float(test_y.size(0))
    # print('test accuracy: %.2f' % accuracy)
    return accuracy,duration


if __name__=='__main__':
    lrs=[0.0005,0.005,0.01,0.05,0.1]
    # epochs=[500,1000,2000,5000]
    momentums=[0.9,0.8,0.7,0.6,]
    batches=[10,20,30]
#     results={}
#     for e in epochs:
#         for b in batches:
#             for m in momentums:
    for b in [20]:
                    print('running 5 times......')
                    accus=[];times=[]
                    for i in range(5):
                        # print('and running......')
                        accuracy,duration=train_and_test(1000,b,0.005,0.8)
                        print('accuracy:',accuracy,'duration:',duration)
                        accus.append(accuracy);times.append(duration)

                    avg_a=np.mean(accus);avg_t=np.mean(times)
                    print('26-40-40-15-3','accuracy:',avg_a,'| duration:',avg_t)
                    # print('lr:',0.005,'| epoch:',1000,'| momentum:',0.7,'| batch size:',b,'accuracy:',avg_a,'| duration:',avg_t)
#     for h in range(5,14):
#         accus = [];times = []
#         for i in range(2):
#             print('running 2 times')
#             accuracy,duration=train_and_test(1000,10,0.01,0.9,h)
#             accus.append(accuracy);times.append(duration)
#             print(accuracy)
#         avg_a=np.mean(accus);avg_t=np.mean(times)
#         print('hidden units:',h,'accuracy:',avg_a,'| duration:',avg_t)





















