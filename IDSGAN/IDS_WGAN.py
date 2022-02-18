
import numpy as np
import pandas as pd
import torch as th
from torch.autograd import Variable as V
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing import preprocess,create_batch2
from model.model_class import Blackbox_IDS,Generator,Discriminator
import matplotlib.pyplot as plt
import adabound 

def compute_gradient_penalty(D, normal_t, attack_t):
    alpha = th.Tensor(np.random.random((normal_t.shape[0], 1)))
    between_n_a = (alpha * normal_t + ((1 - alpha) * attack_t)).requires_grad_(True)
    d_between_n_a = D(between_n_a)
    adv = V(th.Tensor(normal_t.shape[0], 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(
        outputs=d_between_n_a,
        inputs=between_n_a,
        grad_outputs=adv,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

data = pd.read_csv("dataset/half_GAN_KDDTrain+.csv")
train_data,raw_attack,normal,true_label = preprocess(data)
BATCH_SIZE = 256 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10     # Gradient penalty lambda hyperparameter
MAX_EPOCH = 100 # How many generator iterations to train for
D_G_INPUT_DIM = len(train_data.columns)
G_OUTPUT_DIM = len(train_data.columns)
D_OUTPUT_DIM = 1
CLAMP = 0.01

# read parameters of IDS
ids_model = Blackbox_IDS(D_G_INPUT_DIM,2)
param = th.load('save_model/IDS.pth')
ids_model.load_state_dict(param)
#read model
generator = Generator(D_G_INPUT_DIM,G_OUTPUT_DIM)
discriminator = Discriminator(D_G_INPUT_DIM,D_OUTPUT_DIM)

optimizer_G = optim.RMSprop(generator.parameters(), lr=0.0001)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.0001)

batch_attack = create_batch2(raw_attack,BATCH_SIZE)
d_losses,g_losses = [],[]
ids_model.eval()
generator.train()
discriminator.train()
cnt = -5
print("IDSGAN start training")
print("-"*100)
for epoch in range(MAX_EPOCH):

    batch_normal = create_batch2(normal,BATCH_SIZE)
    run_g_loss = 0.
    run_d_loss = 0.
    c=0
    for bn in batch_normal:

        normal_b = th.Tensor(bn)
        #  Train Generator
        for p in discriminator.parameters():  
            p.requires_grad = False
    
        optimizer_G.zero_grad()
        z = V(th.Tensor(raw_attack[np.random.randint(0,len(raw_attack),BATCH_SIZE)]+ np.random.uniform(0,1,(BATCH_SIZE,D_G_INPUT_DIM))))
        adversarial_attack = generator(z)
        D_pred= discriminator(adversarial_attack)
        g_loss = -th.mean(D_pred)
        g_loss.backward()
        optimizer_G.step()
        
        run_g_loss += g_loss.item()
        # Train Discreminator
        for p in discriminator.parameters(): 
            p.requires_grad = True
        
        for c in range(CRITIC_ITERS):
            optimizer_D.zero_grad()
            for p in discriminator.parameters():
                p.data.clamp_(-CLAMP, CLAMP)
                
            z = V(th.Tensor(raw_attack[np.random.randint(0,len(raw_attack),BATCH_SIZE)] + np.random.uniform(0,1,(BATCH_SIZE,D_G_INPUT_DIM))))
            adversarial_attack = generator(z).detach()
            ids_input = th.cat((adversarial_attack,normal_b))

            l = list(range(len(ids_input)))
            np.random.shuffle(l)
            ids_input = V(th.Tensor(ids_input[l]))
            ids_pred = ids_model(ids_input)
            ids_pred_lable = th.argmax(nn.Sigmoid()(ids_pred),dim = 1).detach().numpy()

            pred_normal = ids_input.numpy()[ids_pred_lable==0]
            pred_attack = ids_input.numpy()[ids_pred_lable==1]
            
            print(len(pred_normal))
            if len(pred_attack) == 0:
                cnt += 1
                break
            D_noraml = discriminator(V(th.Tensor(pred_normal)))
            D_attack= discriminator(V(th.Tensor(pred_attack)))
            
            loss_normal = th.mean(D_noraml)
            loss_attack = th.mean(D_attack)
            gradient_penalty = compute_gradient_penalty(discriminator, normal_b.data, adversarial_attack.data)
            
            d_loss =  loss_attack - loss_normal #+ LAMBDA * gradient_penalty

            d_loss.backward()
            optimizer_D.step()
            run_d_loss += d_loss.item()

    d_losses.append(run_d_loss/CRITIC_ITERS)
    g_losses.append(run_g_loss)
    print(f"{epoch} : {run_g_loss} \t {run_d_loss/CRITIC_ITERS}")
    if cnt >= 100:
        print("Not exist predicted attack traffic")
        break
print("IDSGAN finish training")
th.save(generator.state_dict(), 'save_model/generator.pth')
th.save(discriminator.state_dict(), 'save_model/discriminator.pth')
plt.plot(d_losses,label = "D_loss")
plt.plot(g_losses, label = "G_loss")
plt.legend()
plt.show()