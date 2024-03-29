import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from scipy.linalg import eigvalsh
import numpy as np
from scipy.optimize import nnls

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
    s_das = torch.zeros(s_size, N, dtype=H.dtype)
    for i in range(N):
        s_das[:, i] = torch.abs(torch.randn(s_size)).to(torch.int64)
        x[:, i] = H @ s_das[:, i]
        solution, _ = nnls(H, x[:, i])
        s[:, i] = torch.tensor(solution)
    dataset = SimulatedData(x=x, s=s)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return data_loader

b_x_list = []
b_s_list = []

def train(model, train_loader, num_epochs, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    train_loss = []

    for epoch in range(num_epochs):
        model.train()
        batch_train_loss = 0
        for batch, (b_x, b_s) in enumerate(train_loader):
            b_x, b_s = b_x.to(device), b_s.to(device)

            s_hat = model(b_x)
            s_hat = s_hat.t()
            loss = loss_fn(s_hat, b_s)
            batch_train_loss += loss.data.item()

            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            max_grad_norm = 1.0

            for param in model.parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(param, max_grad_norm)

            optimizer.step()

            b_x_list.append(b_x)
            b_s_list.append(b_s)

        train_loss.append(batch_train_loss / len(train_loader))
        if epoch % 20 == 0:
            print("Epoch %d, Train loss %.8f" % (epoch, train_loss[epoch]))

    return train_loss

class NNLS_model(nn.Module):
    def __init__(self, x_size, s_size, L, first=False):
        super(NNLS_model, self).__init__()
        self.x_size = x_size
        self.s_size = s_size
        self.L = L
        self.first = first
        self.A = nn.Linear(s_size, x_size, bias=False)
        self.eta = torch.ones((1, self.A.weight.shape[1]), requires_grad=True)

    def forward(self, s):
        if self.first:
            device = self.A.weight.device
            ones_tensor = torch.ones(s.shape[0], self.A.weight.shape[0], device=device)
            s = (ones_tensor -
                 self.eta * ((self.A.weight @ (self.A.weight.t() @ ((ones_tensor ** self.L).t()) - s.t()) * ((ones_tensor ** (self.L - 1)).t()))))
        else:
            s = s - self.eta * (self.A.weight.t() @ (self.A.weight @ ((s ** self.L).t()) - s.t()) * ((s ** (self.L - 1)).t()))
        return s



class nnls_model(nn.Module):
    def __init__(self, x_size, s_size, L):
        super(nnls_model, self).__init__()
        self.layers = nn.Sequential(
            NNLS_model(x_size, s_size, L, first=True),
            NNLS_model(x_size, s_size, L)
        )

    def forward(self, x):
        for child in self.layers.children():
            setattr(child, 'x', x)
        return self.layers(x)

def unfolded_nnls_apply(x_size, s_size, train_loader, num_epochs, L):
    model = nnls_model(x_size=x_size, s_size=s_size, L=L)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_train = train(model=model, train_loader=train_loader, num_epochs=num_epochs, device=device)
    return loss_train

if __name__ == "__main__":
    x_size, s_size = 1200, 400
    H = torch.randn(x_size, s_size)
    L = 3
    train_loader = create_data_set(H, x_size=x_size, s_size=s_size, N=500, batch_size=128)
    train_loss = unfolded_nnls_apply(x_size, s_size, train_loader, num_epochs=25000, L=L)
    torch.save(nnls_model(x_size=x_size, s_size=s_size, L=L).state_dict(), 'your_model.pth')
