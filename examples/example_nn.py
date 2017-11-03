"""
Example neural network written in pytorch

Copied from the pytorch tutorial with some
minor changes
"""


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_train_set():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../data', train=True,
        download=False, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='../data', train=False,
        download=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def check_net():
    net = MyNet()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    optimizer = optim.Adam(net.parameters())
    optimizer.zero_grad()

    input_vec = Variable(torch.randn(1, 1, 32, 32))
    out = net(input_vec)
    print(out)

    # net.zero_grad()
    # out.backward(torch.randn(1, 10))

    target = Variable(torch.arange(20, 30))
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)

    # net.zero_grad()
    loss.backward()
    optimizer.step()

    print(net.conv1.weight.grad)

    input_vec_2 = Variable(torch.randn(1, 1, 32, 32))
    out_2 = net(input_vec_2)
    target_2 = Variable(torch.arange(20, 30))
    loss_2 = criterion(out_2, target_2)
    loss_2.backward()
    optimizer.step()

    print(net.conv1.weight.grad)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def main():
    net = MyNet()
    train_data, test_data = load_train_set()
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        for i, data in enumerate(train_data, 0):

            images, labels = data
            images, labels = Variable(images), Variable(labels)

            optimizer.zero_grad()
            out = net(images)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            if i % 2000 == 0:
                print(loss)

    # testing
    dataiter = iter(test_data)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    plt.show()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


    out = net(Variable(images))
    _, predicted = torch.max(out.data, 1)

    print('Predicted', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


if __name__ == "__main__":
    main()
