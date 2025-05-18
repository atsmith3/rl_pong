import sys
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class ExpertPolicyDataset(Dataset):

    def __init__(self):
        self.data = []
        with open("./expert_policy.txt", "r") as f:
            for line in f.readlines():
                tokens = line.strip().split(" ")
                features = torch.tensor([float(x) for x in tokens[:-1]],
                                        dtype=torch.float32)
                label = torch.tensor(int(float(tokens[-1])), dtype=torch.long)
                self.data.append((features, label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]


class SLPongAgent:

    def __init__(self, restore=""):
        self.device = torch.accelerator.current_accelerator(
        ).type if torch.accelerator.is_available() else "cpu"
        self.model = NeuralNetwork().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        print(self.model)
        if restore:
            print(f"Restoring weights from {restore}")
            self.model.load_state_dict(torch.load(restore))
            self.model.eval()

    def train(self, batch_size=100, epochs=500, lr=1e-1):
        train_dataloader = DataLoader(ExpertPolicyDataset(),
                                      batch_size=batch_size,
                                      shuffle=True)
        print(f"SLPongAgent: Using {self.device} device")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
        torch.save(model.state_dict(), "pong_agent.pth")
        print("Training completed")

    def eval(self, state) -> int:
        X = torch.tensor([state], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            pred = self.model(X)
            action = pred.argmax(1)
        return int(action.item())


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(5, 512), nn.ReLU(),
                                               nn.Linear(512, 512), nn.ReLU(),
                                               nn.Linear(512, 512), nn.ReLU(),
                                               nn.Linear(512, 3))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    agent = SLPongAgent()
    agent.train(100, 500, 1e-1)
