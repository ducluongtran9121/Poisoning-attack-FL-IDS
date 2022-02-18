import pandas as pd
import numpy as np

def create_batch1(x,y,batch_size):
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    y = y[a]

    batch_x = [x[batch_size * i : (i+1)*batch_size,:].tolist() for i in range(len(x)//batch_size)]
    batch_y = [y[batch_size * i : (i+1)*batch_size].tolist() for i in range(len(x)//batch_size)]
    return batch_x, batch_y
def create_batch2(x,batch_size):
    a = list(range(len(x)))
    np.random.shuffle(a)
    x = x[a]
    batch_x = [x[batch_size * i : (i+1)*batch_size,:] for i in range(len(x)//batch_size)]
    return batch_x
# other than discret variable
def preprocess1(train):

    del train["id"]
    del train["land"]
    del train["protocol_type"]
    del train["logged_in"]
    del train["su_attempted"]
    del train["is_host_login"]
    del train["is_guest_login"]
    del train["flag"]
    del train["service"]

    numeric_columns = list(train.select_dtypes(include=['int',"float"]).columns)
    for c in numeric_columns:
        max_ = train[c].max()
        min_ = train[c].min()
        train[c] = train[c].map(lambda x : (x - min_)/(max_ - min_))

    train["num_outbound_cmds"] = train["num_outbound_cmds"].map(lambda x : 0)


    train["class"] = train["class"].map(lambda x : 1 if x == "anomaly" else 0)

    raw_attack = np.array(train[train["class"] == 1])[:,:-1]
    normal = np.array(train[train["class"] == 0])[:,:-1]
    true_label = train["class"]

    del train["class"]
    return train,raw_attack,normal,true_label
#all
def preprocess2(train,test):

    train["land"] = train["land"].astype("object")
    train["logged_in"] = train["logged_in"].astype("object")
    train["su_attempted"] = train["su_attempted"].astype("object")
    train["is_host_login"] = train["is_host_login"].astype("object")
    train["is_guest_login"] = train["is_guest_login"].astype("object")
    train["is_guest_login"] = train["is_guest_login"].astype("object")

    test["land"] = test["land"].astype("object")
    test["logged_in"] = test["logged_in"].astype("object")
    test["su_attempted"] = test["su_attempted"].astype("object")
    test["is_host_login"] = test["is_host_login"].astype("object")
    test["is_guest_login"] = test["is_guest_login"].astype("object")
    test["is_guest_login"] = test["is_guest_login"].astype("object")


    some_kind = ["su_attempted","protocol_type","flag","service"]
    train = pd.get_dummies(train,columns=some_kind,drop_first=True,sparse=True)
    test = pd.get_dummies(test,columns=some_kind,drop_first=True,sparse=True)

    aol = train["service_aol"]
    trash = list(set(train.columns) - set(test.columns))
    for t in trash:
        del train[t]
    train["service_aol"] = aol
    test["service_aol"] = np.zeros(len(test)).astype("uint8")

        
    train["class"] = train["class"].map(lambda x : 1 if x == "anomaly" else 0)
    test["class"] = test["class"].map(lambda x : 1 if x == "anomaly" else 0)

    train["land"] = train["land"].astype("int")
    train["logged_in"] = train["logged_in"].astype("int")
    train["is_host_login"] = train["is_host_login"].astype("int")
    train["is_guest_login"] = train["is_guest_login"].astype("int")
    train["is_guest_login"] = train["is_guest_login"].astype("int")

    test["land"] = test["land"].astype("int")
    test["logged_in"] = test["logged_in"].astype("int")
    test["is_host_login"] = test["is_host_login"].astype("int")
    test["is_guest_login"] = test["is_guest_login"].astype("int")
    test["is_guest_login"] = test["is_guest_login"].astype("int")

    train["num_outbound_cmds"] = train["num_outbound_cmds"].map(lambda x : 0)
    test["num_outbound_cmds"] = test["num_outbound_cmds"].map(lambda x : 0)

    trainx, trainy = np.array(train[train.columns[train.columns != "class"]]), np.array(train["class"])
    testx, testy= np.array(test[train.columns[train.columns != "class"]]),np.array(test["class"])
    return trainx,trainy,testx,testy