import torch as th
from torch import nn
class Blackbox_IDS(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim*2),
            nn.Dropout(0.6),   
            #nn.ELU(),
            nn.LeakyReLU(True),
           # nn.BatchNorm1d(input_dim*2),
            nn.Linear(input_dim *2, input_dim *2),
            nn.Dropout(0.5),       
           # nn.ELU(),
#            nn.ReLU(True),
            nn.LeakyReLU(True),   
          # nn.BatchNorm1d(input_dim*2),
            nn.Linear(input_dim *2, input_dim//2),
            nn.Dropout(0.5),       
#            nn.ReLU(True),
        #  nn.ELU(),
            nn.LeakyReLU(True),
           #nn.BatchNorm1d(input_dim//2),
            nn.Linear(input_dim//2,input_dim//2),
            nn.Dropout(0.4),       
        #    nn.ELU(),
#            nn.ReLU(True),
            nn.LeakyReLU(True),
            
             nn.Linear(input_dim//2,output_dim),
        )
        #nn.init.kaiming_normal_(self.layer.weight)
        self.output = nn.Sigmoid()
        #self.output = nn.Softmax()
    def forward(self,x):
        x = self.layer(x)
        return x
class Generator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim //2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim // 2, input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,input_dim//2),
            nn.ReLU(True),
            nn.Linear(input_dim//2,output_dim),
        )
    def forward(self,x):
        x = self.layer(x)
        return th.clamp(x,0.,1.)

class Discriminator(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim * 2, input_dim *2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim*2 , input_dim*2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim*2,input_dim//2),
            nn.LeakyReLU(True),
            nn.Linear(input_dim//2,output_dim),
        )

    def forward(self,x):
        return self.layer(x)