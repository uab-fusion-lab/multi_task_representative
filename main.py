import random
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import fetch_dataset, data_to_tensor, iid_partition_loader, noniid_partition_loader
from models import CNN, MLP

# set random seeds
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("| using device:", device)

# hyperparams
bsz = 10
train_data, test_data = fetch_dataset()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)  # inference bsz=100

rhos = 1e-3
clientslist = []

def data_preparation():
    # get client dataloaders
    iid_client_train_loader = iid_partition_loader(train_data, bsz=bsz)
    noniid_client_train_loader = noniid_partition_loader(train_data, bsz=bsz)

    # # iid
    # label_dist = torch.zeros(10)
    # for (x, y) in iid_client_train_loader[25]:
    #     label_dist += torch.sum(F.one_hot(y, num_classes=10), dim=0)
    #
    #
    # # non-iid
    # label_dist = torch.zeros(10)
    # for (x, y) in noniid_client_train_loader[25]:
    #     label_dist += torch.sum(F.one_hot(y, num_classes=10), dim=0)

    return iid_client_train_loader, noniid_client_train_loader

criterion = nn.CrossEntropyLoss()

def validate(model):
    model = model.to("cuda")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (t, (x,y)) in enumerate(test_loader):
            x = x.to("cuda")
            y = y.to("cuda")
            out = model(x)
            correct += torch.sum(torch.argmax(out, dim=1) == y).item()
            total += x.shape[0]
    return correct/total

def train_client(id, client_loader, global_model, num_local_epochs, lr):
    # local_model = copy.deepcopy(global_model)
    local_model = clientslist[id]
    local_model = local_model.to("cuda")

    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    local_model.model.load_state_dict(global_model.model.state_dict())
    for epoch in range(num_local_epochs):
        for (i, (x,y)) in enumerate(client_loader):
            x = x.to("cuda")
            y = y.to("cuda")
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

    return local_model


def fed_avg_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds,
                       filename):
    round_accuracy = []
    for i in range(10):
        clientslist.append(copy.deepcopy(global_model))
    for t in range(max_rounds):
        print("starting round {}".format(t))

        # choose clients
        clients = np.random.choice(np.arange(50), num_clients_per_round, replace=False)
        print("clients: ", clients)

        global_model.eval()
        global_model = global_model.to("cuda")
        running_avg = None
        u_avg = None


        for i, c in enumerate(clients):
            # train local client
            print("round {}, starting client {}/{}, id: {}".format(t, i + 1, num_clients_per_round, c))
            local_model = train_client(i, client_train_loader[c], global_model, num_local_epochs, lr)

            # add local model parameters to running average
            running_avg = running_model_avg(running_avg, local_model.model.state_dict(), 1 / num_clients_per_round)
            u_avg = running_u_avg(u_avg, local_model, 1 / num_clients_per_round)

        # set global model parameters for the next step
        global_model.model.load_state_dict(running_avg)


        # validate
        val_acc = validate(global_model)
        print("round {}, validation acc: {}".format(t, val_acc))
        round_accuracy.append(val_acc)

        if (t % 10 == 0):
            np.save(filename + '_{}'.format(t) + '.npy', np.array(round_accuracy))

    return np.array(round_accuracy)

def running_model_avg(current, next, scale):
    if current == None:
        current = next
        for key in current:
            current[key] = current[key] * scale
    else:
        for key in current:
            current[key] = current[key] + (next[key] * scale)
    return current

def running_u_avg(current, model, scale):
    if current == None:
        current = {key: value * scale for key, value in model.ADMM_U.items()}
    else:
        for key in current:
            current[key] = current[key] + (model.ADMM_U[key] * scale)
    return current

def main():
    iid_client_train_loader, noniid_client_train_loader = data_preparation()
    mlp = CNN()
    mlp_iid_m10 = copy.deepcopy(mlp)
    acc_mlp_iid_m10 = fed_avg_experiment(mlp_iid_m10, num_clients_per_round=10,
                                         num_local_epochs=1,
                                         lr=0.05,
                                         client_train_loader=iid_client_train_loader,
                                         max_rounds=100,
                                         filename='./acc_mlp_iid_m10')
    print(acc_mlp_iid_m10)
    np.save('cifarresult/acc_cnn_admm_noniid_m10.npy', acc_mlp_iid_m10)


if __name__ == '__main__':
    main()