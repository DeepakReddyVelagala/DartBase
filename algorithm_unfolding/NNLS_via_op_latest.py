import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from scipy.linalg import eigvalsh
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import torch.optim.lr_scheduler as lr_scheduler
torch.set_default_dtype(torch.float64)


class SimulatedData(Data.Dataset):
    def __init__(self, x, s):
        self.x = x
        self.s = s

    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        x = self.x[:, idx]
        s = self.s[:, idx]
        return x, s


def create_data_set(H, x_size, s_size, N, batch_size):
    x = torch.zeros(x_size, N)
    s = torch.zeros(s_size, N)
    # Generating the dataset using NNLS solver from scipy
    s_das = torch.zeros(s_size, N, dtype=H.dtype)
    for i in range(N):
        s_das[:,i] = torch.abs(torch.randn(s_size)).to(torch.int64) #Non negative vectors
#         s_das[:, i] = torch.randn(s_size).to(torch.int64) #Random vectors
        x[:, i] = H @ s_das[:, i]
        solution, _ = nnls(H, x[:, i])
        s[:, i] = torch.tensor(solution)
    dataset = SimulatedData(x=x, s=s)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

b_x_list = []
b_s_list = []

def train(model, train_loader, num_epochs):
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.0009, momentum=0.99, weight_decay=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #0.00009
#     initial_lr, factor, patience, min_lr=0.0005, 0.1, 10, 1e-7
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
    loss_fn = nn.MSELoss()
    train_loss = []
    
    for epoch in range(num_epochs):
        model.train()
        batch_train_loss = 0
        for batch, (b_x, b_s) in enumerate(train_loader):
            s_hat = model(b_x)
            s_hat = s_hat.t()
            loss = loss_fn(s_hat, b_s)  #The loss is dancing around 0.015
#             loss = loss_fn(H@(s_hat.t()), H@(b_s.t()))
            batch_train_loss += loss.data.item()
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            max_grad_norm = 1.0
            # Perform gradient clipping during training
            for param in model.parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(param, max_grad_norm)
            optimizer.step()
            b_x_list.append(b_x)
            b_s_list.append(b_s)
        train_loss.append(batch_train_loss / len(train_loader))
        if epoch % 20 == 0:
              print("Epoch %d, Train loss %.8f"
                  % (epoch, train_loss[epoch]))
#         scheduler.step(train_loss[epoch])
    return train_loss


class NNLS_model(nn.Module):
    def __init__(self, x_size, s_size, L, first=False):
        super(NNLS_model, self).__init__()
        self.x_size = x_size
        self.s_size = s_size
        self.L = L
        self.first = first
        self.A = nn.Linear(s_size, x_size, bias=False)
        self.eta = torch.ones((self.A.weight.shape[1], 1), requires_grad=True)

    def forward(self, s):
        if self.first:
            s = (torch.ones(self.x.shape[0], self.s_size).t()-self.eta * ((self.A.weight.t() @ (self.A.weight @ ((torch.ones(self.x.shape[0], self.s_size) ** (self.L)).t())- self.x.t()) * ((torch.ones(self.x.shape[0], self.s_size) ** ((self.L) - 1)).t()))))
        else:
            s = s - self.eta * (self.A.weight.t() @ (self.A.weight @ ((s ** (self.L))) - self.x.t()) * ((s ** ((self.L) - 1))))
        return s


class nnls_model(nn.Module):
    def __init__(self, x_size, s_size, L):
        super(nnls_model, self).__init__()
        self.layers = nn.Sequential(
            NNLS_model(x_size, s_size, L, first=True),
            NNLS_model(x_size, s_size, L),
#             NNLS_model(x_size, s_size, L),
#             NNLS_model(x_size, s_size, L),
#             NNLS_model(x_size, s_size, L)
        )

    def forward(self, x):
        for child in self.layers.children():
            setattr(child, 'x', x)
        return self.layers(x)


def unfolded_nnls_apply(x_size, s_size, train_loader, num_epochs, L):
    model = nnls_model(x_size=x_size, s_size=s_size, L=L)
    loss_train = train(model=model, train_loader=train_loader, num_epochs=num_epochs)
    return loss_train


if __name__ == "__main__":
    x_size, s_size = 1200, 400 #150, 200
    H = torch.randn(x_size, s_size)
    L = 3
#     Binary matrix
#     H = generate_binary_matrix(x_size, s_size) #torch.randn(x_size, s_size)
#     H = 1*(H / torch.norm(H, p='fro'))
    train_loader = create_data_set(H, x_size=x_size, s_size=s_size, N=500, batch_size=250)
    # test_loader = create_data_set(H, x_size=x_size, s_size=s_size, N=40, k=15, batch_size=8)
    # re = ista(H, test_loader.dataset.x[:,0],0.001,1000 )
    train_loss = unfolded_nnls_apply(x_size, s_size, train_loader, num_epochs=25000, L=L)
    torch.save(nnls_model(x_size=x_size, s_size=s_size, L=L).state_dict(), 'your_model.pth')