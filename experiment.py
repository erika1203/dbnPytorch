import numpy as np
# from RBM import *
from DBN import *
import torch



def data_process(filename):     #数据归一化
    fr = open(filename, 'r')
    content = fr.readlines()
    datas = [];i=j=l=0
    for x in content[1:]:
        k=content[1:].index(x)
        x = x.strip().split(' ')
        datas.append([float(i) for i in x[:-2]])
        if x[-1]=='危险':
            datas[k].append(0);i+=1
        elif x[-1]=='可疑':
            datas[k].append(1);j+=1
        elif x[-1]=='健康':
            datas[k].append(2)
            l+=1
    print('0:', i,'1:',j,'2:',l)
    # print(datas)
    for dat in datas:
        if l>233 and dat[-1]==2:
            del(dat);l-=1
    print('0:', i, '1:', j, '2:', l)
    datas = np.array(datas).astype('float32')
    m,n=datas.shape

    maxMat=np.max(datas,0)
    minMat=np.min(datas,0)
    diffMat=maxMat-minMat
    for j in range(n-1):
        datas[:,j]=(datas[:,j]-minMat[j])/diffMat[j]
    return datas

def gen_train_test(dat):        #分割训练集和测试集
    np.random.shuffle(dat)
    traind=torch.from_numpy(dat[:500,:-1]).float()
    trainl=torch.from_numpy(dat[:500,-1]).long()
    testd = torch.from_numpy(dat[500:,:-1]).float()
    testl = torch.from_numpy(dat[500:,-1]).long()
    # def filtering(tensor):
    #     for i in range(len(tensor)):
    #             if tensor[i] == 0:
    #                 tensor[i] = -1
    #             elif tensor[i] == 1:
    #                 tensor[i] = 0
    #             else:
    #                 tensor[i] = 1
    #     return tensor
    #
    # # 把标签值转化为二元(0/1)
    # trainl[trainl == 0] = -1
    # trainl[trainl== 1] = 0
    # trainl[trainl == 2] = 1
    # trainl = filtering(trainl)
    #
    # testl[testl == 0] = -1
    # testl[testl == 1] = 0
    # testl[testl == 2] = 1
    # testl = filtering(testl)
    # # print(trainl[0])
    # print(dat[0])
    return traind,trainl,testd,testl

def train_batch(traind,trainl,testd,testl,SIZE=20,SHUFFLE=True,WORKER=2):   #分批处理
    trainset=torch.utils.data.TensorDataset(traind,trainl)
    trainloader=torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=SIZE,
        shuffle=SHUFFLE,
        num_workers=WORKER,  )
    testset = torch.utils.data.TensorDataset(testd, testl)
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=SIZE,
        shuffle=SHUFFLE,
        num_workers=WORKER, )
    return trainloader,testloader

dat=data_process('data.txt')
traind,trainl,testd,testl=gen_train_test(dat)
# print(dat[0])
# print(traind);print(trainl)
train_loader,test_loader=train_batch(traind,trainl,testd,testl,20,SHUFFLE=True,WORKER=2)
# print(len(train_loader),len(test_loader))

# a,b=train_and_test(traind,trainl,testd,testl,train_loader)

# rbm=RBM()
# rbm.trains(train_loader)
 # rbm.extract_features(test_loader)




