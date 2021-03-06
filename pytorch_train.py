import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from torchvision import datasets, transforms

from tqdm import tqdm

if not torch.cuda.is_available(): 
    print('No CUDA device')
    exit()

dev = torch.device("cuda")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=64, shuffle=False)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

model.to(dev)
opt = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for xb, yb in tqdm(train_loader):
        pred = model(xb.to(dev))
        loss = loss_fn(pred, yb.to(dev))

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(sum(model(xb.to(dev)).argmax(-1) == yb.to(dev)) for xb, yb in tqdm(test_loader))

    print(epoch, valid_loss / (len(test_loader) * 64.))