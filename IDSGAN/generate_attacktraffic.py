from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from preprocessing import preprocess2,create_batch2
from model.model_class import Blackbox_IDS,Generator
import torch as th
from torch import nn
test = pd.read_csv("dataset/preproKDDTest+.csv")
test_raw_attack = np.array(test[test["class"] == 1])[:,:-1]
test_normal = np.array(test[test["class"] == 0])[:,:-1]
true_label = test["class"]
BATCH_SIZE = 256 # Batch size
D_G_INPUT_DIM = test_normal.shape[1]
G_OUTPUT_DIM =test_normal.shape[1] 
D_OUTPUT_DIM = 1

#read model
random_g = Generator(D_G_INPUT_DIM,G_OUTPUT_DIM)
leaned_g = Generator(D_G_INPUT_DIM,G_OUTPUT_DIM)

ids_model = Blackbox_IDS(D_G_INPUT_DIM,2)

ids_param= th.load('save_model/20__IDS.pth',map_location=lambda x,y:x)
ids_model.load_state_dict(ids_param)
g_param = th.load('save_model/generator.pth',map_location=lambda x,y:x)
leaned_g.load_state_dict(g_param)

model_g = {"no_learn":random_g,"leaned":leaned_g}

test_batch_normal = create_batch2(test_normal,BATCH_SIZE)

print("adversarial traffic evaluating")
print("-"*100)
for n,g in model_g.items():
    o_dr,a_dr,eir=[],[],[]
    g.eval()
    with th.no_grad():
        for bn in test_batch_normal:
            normal_b = th.Tensor(bn)
            batch_a= th.Tensor(test_raw_attack[np.random.randint(0,len(test_raw_attack),BATCH_SIZE)])
            z = batch_a + th.Tensor(np.random.uniform(0,1,(BATCH_SIZE,D_G_INPUT_DIM)))
            
            adversarial_attack = g(z)
            adversarial_attack[:,33:] = th.Tensor(np.where(adversarial_attack[:,33:].detach().cpu().numpy()>= 0.5 , 1,0))
            ori_input = th.cat((batch_a,normal_b))
            adv_input = th.cat((adversarial_attack,normal_b))
            l = list(range(len(ori_input)))
            np.random.shuffle(l)
            
            adv_input = adv_input[l]
            ori_input = ori_input[l]
            ids_pred_adv = ids_model(adv_input)
            ids_pred_ori = ids_model(ori_input)
            
            ids_true_label = np.r_[np.ones(BATCH_SIZE),np.zeros(BATCH_SIZE)][l]
            pred_label_adv = th.argmax(nn.Sigmoid()(ids_pred_adv),dim = 1).cpu().numpy()
            pred_label_ori = th.argmax(nn.Sigmoid()(ids_pred_ori),dim = 1).cpu().numpy()
            
            
            tn1, fp1, fn1, tp1 = confusion_matrix(ids_true_label,pred_label_adv).ravel()
            tn2, fp2, fn2, tp2 = confusion_matrix(ids_true_label,pred_label_ori).ravel()
            o_dr.append(tp2/(tp2 + fp2))
            a_dr.append(tp1/(tp1 + fp1))
            eir.append(1 - (tp1/(tp1 + fp1))/(tp2/(tp2 + fp2)))
    print(f"{n} => origin_DR : {np.mean(o_dr)} \t advasarial_DR : {np.mean(a_dr)} \t EIR : {np.mean(eir)}")   