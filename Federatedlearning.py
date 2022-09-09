import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Activation, BatchNormalization, LSTM
from keras.optimizers import adam_v2
from keras import backend as K
from preprocessing import preprocessing_72cols, preprocessing_GAN, _preprocessing
import argparse
np.set_printoptions(precision=3, suppress=True)
# Set a fixed seed
tf.random.set_seed(100)
np.random.seed(100)

parser = argparse.ArgumentParser("Training FL with defined clients")
parser.add_argument('--num_clients', '-n', help='Number of clients in FL', type=int, default=10)
parser.add_argument('--num_attackers', '-a', help='Number of attackers in FL', type=int, default=4)
args = parser.parse_args()
NUM_CLIENTS = args.num_clients
NUM_ATTACKER = args.num_attackers

print(f"{'-'*25}Setup environment sucessfully{'-'*25}")

client_names = []
clients = {}


for i in range(1, NUM_CLIENTS+1):
    data = pd.read_csv(f"/home/haochu/Documents/PoisoningAttack/Dataset/N-BaIoT/Original/{NUM_CLIENTS}Clients/Origin/Client{i}.csv")
    X, y = preprocessing_72cols(data)
    X = np.reshape(X, (len(X), X.shape[1], 1))
    client_names.append('{}_{}'.format('client', i))
    clients[client_names[-1]] = list(zip(X, y))

print(client_names)

print("\n --- Load data normal successfully ---")

attackers = {}
attackers_name = []
list_attackers = [1,3,5,7][:NUM_ATTACKER]

full_clients = list(range(1,NUM_CLIENTS+1))
normal_clients = list(set(full_clients)-set(list_attackers))

for i in list_attackers:
    data = pd.read_csv(f"/home/haochu/Documents/PoisoningAttack/Dataset/N-BaIoT/Original/{NUM_CLIENTS}Clients/Origin/Client{i}.csv")
    X, y = preprocessing_72cols(data)
    X = np.reshape(X, (len(X), X.shape[1], 1))
    y = y.to_numpy()
    y_new = np.array([abs(s-1) for s in y])
    attackers_name.append('{}_{}'.format('client', i))
    attackers[attackers_name[-1]] = list(zip(X, y_new))

for i in normal_clients:
    data = pd.read_csv(f"/home/haochu/Documents/PoisoningAttack/Dataset/N-BaIoT/Original/{NUM_CLIENTS}Clients/Origin/Client{i}.csv")
    X, y = preprocessing_72cols(data)
    X = np.reshape(X, (len(X), X.shape[1], 1))
    attackers_name.append('{}_{}'.format('client', i))
    attackers[attackers_name[-1]] = list(zip(X, y))

print(attackers_name)
print("\n --- Load data attack successful ---")

X_test = pd.read_csv(f"/home/haochu/Documents/PoisoningAttack/Dataset/N-BaIoT/Original/10Clients/Test.csv")
X_test, y_test = preprocessing_72cols(X_test)
X_test = np.reshape(X_test, (len(X_test), X_test.shape[1], 1))



def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensors((list(data), list(label)))
    return dataset.batch(bs)


print(f"{'-'*25}Preprocessing data sucessfully{'-'*25}")


def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clients
    global_count = sum([tf.data.experimental.cardinality(
        clients_trn_data[cl_name]).numpy() for cl_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(
        clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def test_model(X_test, Y_test, model):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    TN1, FP1, FN1, TP1 = confusion_matrix(Y_test, y_pred).ravel()
    acc = (TP1+TN1)/(TN1+FP1+FN1+TP1)
    pre = TP1/(TP1+FP1)
    rec = TP1/(TP1+FN1)
    f1 = 2*TP1/(2*TP1+FP1+FN1)
    return acc, pre, rec, f1


# hyper-parameter
batch_size = 128
num_classes = 2
epochs = 4
round = 12
filter_size = 3
droprate = 0.50
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [tf.keras.metrics.BinaryAccuracy(),
           tf.keras.metrics.Recall(),
           tf.keras.metrics.Precision()]

# Optimizer
lr_client = 0.0001
lr_server = 0.01
optimizer_client = adam_v2.Adam(learning_rate=lr_client)
optimizer_server = adam_v2.Adam(learning_rate=lr_server)

# Defining CNN Global Model

model = Sequential()

# convolution 1st layer
model.add(Conv1D(64, kernel_size=(filter_size), padding="same",
                 activation='relu',
                 input_shape=(115, 1)))
model.add(BatchNormalization())
# model.add(Dropout(droprate))

# convolution 2nd layer
model.add(Conv1D(128, kernel_size=(filter_size), activation='relu'))
model.add(BatchNormalization())
# model.add(MaxPooling1D(strides=1))
# model.add(Dropout(droprate))

# # convolution 3rd layer
model.add(Conv1D(256, kernel_size=(filter_size), activation='relu'))
model.add(BatchNormalization())
# model.add(MaxPooling1D(strides=1))
# model.add(Dropout(droprate))

# FCN 1st layer
model.add(Flatten())
model.add(Dense(64, use_bias=False))
model.add(Dense(16, use_bias=False))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=['accuracy'])

global_model = model

# input = tf.keras.Input(shape=(67, 1))
# x = tf.keras.layers.LSTM(64)(input)
# x = tf.keras.layers.Dense(16)(x)
# output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
# global_model = tf.keras.Model(inputs=[input], outputs=[output])


print(f"{'-'*25}Federated Learning training phase{'-'*25}")
# print(attackers_name[:NUM_ATTACKER])
print("--------attack----------")
print(attackers_name)
for comm_round in range(1, round+1):
    print("[+] Round ", comm_round)
    start = datetime.datetime.now().replace(microsecond=0)
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    clients_batched = {}
    # num_samples = len(data)
    # if comm_round < 6:
    #     num_samples = len(data) // 3
    # else:
    #     num_samples = len(data) // round * comm_round

    for client_name in client_names:
        if comm_round > 2 and client_name in attackers_name[:NUM_ATTACKER]: # poisoning from round 5
            data = attackers[client_name]
            num_samples = len(data) // round 
            clients_batched[client_name] = batch_data(data[num_samples*(comm_round-1):num_samples*comm_round], num_samples)
        else:
            data = clients[client_name]
            num_samples = len(data) // round 
            clients_batched[client_name] = batch_data(data[num_samples*(comm_round-1):num_samples*comm_round], num_samples)
            # clients_batched[client_name] = batch_data(data[:num_samples], num_samples)

    # loop through each client and create new local model
    for client in client_names:

        # local_model = tf.keras.Model(inputs=[input], outputs=[output])
        local_model = model

        local_model.compile(loss=loss,
                            optimizer=optimizer_client,
                            metrics=metrics)

        # set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        x = next(iter(clients_batched[client]))[0][0].numpy()
        y = next(iter(clients_batched[client]))[1][0].numpy()
        print(f"{client}:{x.shape}")
        # fit local model with client's data
        local_model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

        # scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)

        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)

        scaled_local_weight_list.append(scaled_weights)

        # clear session to free memory after each communication round
        K.clear_session()

    # to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    global_model.set_weights(average_weights)
    print(f"Test set: {X_test.shape}")
    acc, pre, rec, f1 = test_model(X_test, y_test,global_model)
    print(f"Accuracy: {acc:.4f} - Precision: {pre:.4f} - Recall: {rec:.4f} - F1Score: {f1:.4f}")
    end = datetime.datetime.now().replace(microsecond=0)
    print(f"Round {comm_round} finished in: {end-start}")

global_model.save('./Poisoning-FL/CNN/KB1_GANPoisoning')
