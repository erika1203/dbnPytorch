import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


BATCH_SIZE = 20

class RBM(nn.Module):
    '''
    这个类定义了RBM需要的所有函数
    激活函数 : sigmoid
    '''

    def __init__(self,visible_units=26,
                hidden_units = 9,
                k=2,
                learning_rate=1e-3,
                momentum_coefficient=0.5,
                weight_decay = 1e-4,
                use_gpu = False,
                _activation='sigmoid'):
        '''
        定义模型
        W:权重矩阵 shape=(可视节点数,隐藏节点数)
        c:隐藏层偏置 shape=(隐藏节点数 , )
        b : 可视层偏置 shape=(可视节点数 ,)
        '''
        super(RBM,self).__init__()

        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_gpu = use_gpu
        self._activation = _activation

        self.weight = torch.randn(self.visible_units,self.hidden_units) / math.sqrt(self.visible_units)  #初始化
        self.c = torch.randn(self.hidden_units) / math.sqrt(self.hidden_units)
        self.b = torch.randn(self.visible_units) / math.sqrt(self.visible_units)

        self.W_momentum = torch.zeros(self.visible_units,self.hidden_units)
        self.b_momentum = torch.zeros(self.visible_units)
        self.c_momentum = torch.zeros(self.hidden_units)


    def activation(self,X):
        if self._activation=='sigmoid':
            return nn.functional.sigmoid(X)
        elif self._activation=='tanh':
            return nn.functional.tanh(X)
        elif self._activation=='relu':
            return nn.functional.relu(X)
        else:
            raise ValueError("Invalid Activation Function")


    def to_hidden(self ,X):
        '''
        根据可视层生成隐藏层
        通过采样进行
        X 为可视层概率分布
        :param X: torch tensor shape = (n_samples , n_features)
        :return -  hidden - 新的隐藏层 (概率)
                    sample_h - 吉布斯样本 (1 or 0)
        '''
        # print('hinput:',X)
        hidden = torch.matmul(X,self.weight)
        hidden = torch.add(hidden, self.c)  #W.x + c
        # print('mm:',hidden)
        hidden  = self.activation(hidden)

        sample_h = self.sampling(hidden)
        # print('h:',hidden,'sam_h:',sample_h)

        return hidden,sample_h

    def to_visible(self,X):
        '''
        根据隐藏层重构可视层
        也通过采样进行
        X 为隐藏层概率分布
        :returns - X_dash - 新的重构层(概率)
                    sample_X_dash - 新的样本(吉布斯采样)

        '''
        # 计算隐含层激活，然后转换为概率
        # print('vinput:',X)
        X_dash = torch.matmul(X ,self.weight.transpose( 0 , 1) )
        X_dash = torch.add(X_dash , self.b)     #W.T*x+b
        # print('mm:',X_dash)
        X_dash = self.activation(X_dash)

        sample_X_dash = self.sampling(X_dash)
        # print('v:',X_dash, 'sam_v:', sample_X_dash)

        return X_dash,sample_X_dash

    def sampling(self,s):
        '''
        通过Bernoulli函数进行吉布斯采样
        '''
        s = torch.distributions.Bernoulli(s)
        return s.sample()

    def reconstruction_error(self , data):
        '''
        通过损失函数计算重构误差
        '''
        return self.contrastive_divergence(data, False)


    def contrastive_divergence(self, input_data ,training = True):
        '''
        对比散列算法
        '''
        # positive phase
        positive_hidden_probabilities,positive_hidden_act  = self.to_hidden(input_data)

        # 计算 W
        positive_associations = torch.matmul(input_data.t() , positive_hidden_act)



        # negetive phase
        hidden_activations = positive_hidden_act
        for i in range(self.k):     #采样步数
            visible_p , _ = self.to_visible(hidden_activations)
            hidden_probabilities,hidden_activations = self.to_hidden(visible_p)

        negative_visible_probabilities = visible_p
        negative_hidden_probabilities = hidden_probabilities

        # 计算 W
        negative_associations = torch.matmul(negative_visible_probabilities.t() , negative_hidden_probabilities)


        # 更新参数
        if(training):
            self.W_momentum *= self.momentum_coefficient
            self.W_momentum += (positive_associations - negative_associations)

            self.b_momentum *= self.momentum_coefficient
            self.b_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

            self.c_momentum *= self.momentum_coefficient
            self.c_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

            batch_size = input_data.size(0)

            self.weight += self.W_momentum * self.learning_rate / BATCH_SIZE
            self.b += self.b_momentum * self.learning_rate / BATCH_SIZE
            self.c += self.c_momentum * self.learning_rate / BATCH_SIZE

            self.weight -= self.weight * self.weight_decay  # L2 weight decay

        # 计算重构误差
        error = torch.mean(torch.sum((input_data - negative_visible_probabilities)**2 , 1))
        # print('i:',input_data,'o:',negative_hidden_probabilities)

        return error


    def forward(self,input_data):
        return  self.to_hidden(input_data)

    def step(self,input_data):
        '''
            包括前馈和梯度下降，用作训练
        '''
        # print('w:',self.weight);print('b:',self.b);print('c:',self.c)
        return self.contrastive_divergence(input_data , True)


    def trains(self,train_data,num_epochs = 50,batch_size= 20):

        BATCH_SIZE = batch_size

        if(isinstance(train_data ,torch.utils.data.DataLoader)):
            train_loader = train_data
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)


        for epochs in range(num_epochs):
            epoch_err = 0.0

            for batch,_ in train_loader:
            #     batch = batch.view(len(batch) , self.visible_units)

                if(self.use_gpu):
                    batch = batch.cuda()
                batch_err = self.step(batch)

                epoch_err += batch_err


            print("Epoch Error(epoch:%d) : %.4f" % (epochs , epoch_err))
        return


    def extract_features(self,test_dataset):
        if(isinstance(test_dataset ,torch.utils.data.DataLoader)):
            test_loader = test_dataset
        else:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # print(len(test_dataset));print(self.hidden_units)
        test_features = []
        test_labels = []

        for i, (batch, labels) in enumerate(test_loader):
            batch = batch.view(len(batch), self.visible_units)

            if self.use_gpu:
                batch = batch.cuda()

            print(self.to_hidden(batch));print(type(self.to_hidden(batch)));print(type(self.to_hidden(batch)[0]))
            print(labels);print(type(labels))
            # print(BATCH_SIZE,len(batch))
            test_labels.append(labels.numpy())
            test_features.append((self.to_hidden(batch)[0].numpy(),self.to_hidden(batch)[1].numpy()))
            # test_features.append([(m,n) for m,n in zip([x[i] for x in self.to_hidden(batch)[0].numpy()],[x[i] for x in self.to_hidden(batch)[1].numpy()])])
        # print(test_labels[0])
        # print(test_features[0])

        return np.array(test_features),np.array(test_labels)
