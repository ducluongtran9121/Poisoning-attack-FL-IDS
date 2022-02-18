import pandas as pd
import numpy as np
import adabound
import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
from preprocessing import  preprocess2,create_batch1
from model.model_class import Blackbox_IDS
import matplotlib.pyplot as plt
     
train = pd.read_csv("dataset/KDDTrain+.csv")
test = pd.read_csv("dataset/KDDTest+.csv")
trainx,trainy,testx,testy = preprocess2(train,test)
input_dim = trainx.shape[1]
output_dim = 2
batch_size = 1024
tr_N = len(trainx)
te_N = len(testx)
ids_model = Blackbox_IDS(input_dim,output_dim)
opt = optim.Adam(ids_model.parameters(),lr=0.001)
loss_f = nn.CrossEntropyLoss()
max_epoch = 50
train_losses, test_losses = [],[]

def train(x,y):
    ids_model.train()
    batch_x, batch_y = create_batch1(x,y,batch_size)
    run_loss = 0
    for x,y in zip(batch_x,batch_y):

        ids_model.zero_grad()
        x = V(th.Tensor(x),requires_grad = True)
        y = V(th.LongTensor(y))
        out = ids_model(x)
        loss = loss_f(out,y)

        run_loss += loss.item()
        loss.backward()
        opt.step()
    return run_loss/tr_N

def test(x,y):
    ids_model.eval()
    batch_x, batch_y = create_batch1(x,y,batch_size)
    run_loss = 0
    
    with th.no_grad():
        for x,y in zip(batch_x,batch_y):
            x = th.Tensor(x)
            y = th.LongTensor(y)
            out = ids_model(x)
            loss = loss_f(out,y)
            run_loss += loss.item()
    return run_loss/te_N
def main():
    print("IDS start training")
    print("-"*100)
    for epoch in range(max_epoch):
        train_loss = train(trainx,trainy)
        test_loss = test(testx,testy)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"{epoch} : {train_loss} \t {test_loss}")
        
    print("IDS finished training")

    th.save(ids_model.state_dict(), 'model/IDS.pth')
    plt.plot(train_losses,label = "train")
    plt.plot(test_losses, label = "test")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()