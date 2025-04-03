import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(root="data",train=True,transform=ToTensor(),download=True)
testing_data = datasets.FashionMNIST(root="data",train=False,transform=ToTensor(),download=True)

training_dataloader = DataLoader(training_data,batch_size=64)
testing_dataloader = DataLoader(testing_data,batch_size=64)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu = nn.Sequential(
        nn.Linear(28*28,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )

  def forward(self,x):
    x = self.flatten(x)
    logits = self.linear_relu(x)
    return logits

model = NeuralNet().to(device)

from os import device_encoding
# Training the model

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(),lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (x,y) in enumerate(dataloader):
    x,y = x.to(device), y.to(device)

    # compute prediction error
    pred = model(x)
    loss = loss_fn(pred,y)

    # backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch%100 == 0:
      loss, current = loss.item(), (batch+1) * len(x)
      print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)

  model.eval()
  test_loss, correct = 0,0

  with torch.no_grad():
    for x, y in dataloader:
      x,y = x.to(device), y.to(device)

      pred = model(x)
      test_loss += loss_fn(pred,y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_dataloader, model, loss_fn, opt)
    test(testing_dataloader, model, loss_fn)
print("Done!")
