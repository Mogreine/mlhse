import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.datasets import make_blobs, make_moons

import matplotlib.pyplot as plt
from src.hw5.hw5 import MLPClassifier, Linear, ReLU


def test1():
    p = MLPClassifier([
        Linear(4, 64),
        ReLU(),
        Linear(64, 64),
        ReLU(),
        Linear(64, 2)
    ])

    X = np.random.randn(50, 4)
    y = np.array([(0 if x[0] > x[2] ** 2 or x[3] ** 3 > 0.5 else 1) for x in X])
    p.fit(X, y)
    acc = np.mean(p.predict(X).flatten() == y)
    print("Accuracy", acc)


def test2():
    X, y = make_moons(400, noise=0.075)
    X_test, y_test = make_moons(400, noise=0.075)

    best_acc = 0
    for _ in range(10):
        p = MLPClassifier([
                Linear(X.shape[1], 64),
                ReLU(),
                Linear(64, 64),
                ReLU(),
                Linear(64, 2)
            ],
            epochs=10,
            alpha=0.01)

        p.fit(X, y, batch_size=1)
        best_acc = max(np.mean(p.predict(X_test).flatten() == y_test), best_acc)
    print("Accuracy", best_acc)


def test3():
    X, y = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
    X_test, y_test = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
    best_acc = 0
    for _ in range(10):
        p = MLPClassifier([
            Linear(X.shape[1], 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 3)
        ],
            epochs=10,
            alpha=0.01)

        p.fit(X, y, batch_size=1)
        best_acc = max(np.mean(p.predict(X_test).flatten() == y_test), best_acc)
    print("Accuracy", best_acc)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test4():
    model = torchvision.models.resnet18(pretrained=True)
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)

    prediction = model(data)  # forward pass

    loss = (prediction - labels).sum()
    loss.backward()  # backward pass

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()  # gradient descent


def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: Model):
    criterion = nn.CrossEntropyLoss()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    return loss


def train(train_loader, test_loader, model, epochs=100):
    optimizer = torch.optim.Adam(model.parameters())
    train_losses = []
    test_losses = []
    for i in range(epochs):
        #Train
        loss_mean = 0
        elements = 0
        for X, y in iter(train_loader):
            # X = X.to(device)
            # y = y.to(device)
            loss = calculate_loss(X, y, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_mean += loss.item() * len(X)
            elements += len(X)
        train_losses.append(loss_mean / elements)
        #Test
        loss_mean = 0
        elements = 0
        for X, y in iter(test_loader):
            # X = X.to(device)
            # y = y.to(device)
            loss = calculate_loss(X, y, model)
            loss_mean += loss.item() * len(X)
            elements += len(X)
        test_losses.append(loss_mean / elements)
        print("Epoch", i, "| Train loss", train_losses[-1], "| Test loss", test_losses[-1])
    return train_losses, test_losses


def test5():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Model()

    train_l, test_l = train(trainloader, testloader, net)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_l)), train_l, label="train")
    plt.plot(range(len(test_l)), test_l, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # test2()
    # test3()
    test5()
