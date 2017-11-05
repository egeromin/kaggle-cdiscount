"""
Net for training on CDiscount data. Use resnet34 using 
"""
import logging
from argparse import ArgumentParser

import torchvision
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

from dataset import CDiscountDataset
from settings import setup_logging

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def make_model():
    model = torchvision.models.resnet34(pretrained=False, num_classes=5270)
    return model


def train(path_params):
    """
    Train the resnet from scratch using cdiscount data
    :param path_params: The path where to save the params
    """
    dataset = CDiscountDataset('data/cdiscount', train=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)

    model = make_model()
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train a single epoch

    losses = []
    for i, data in enumerate(dataloader, 1):

        image, label = data
        image, label = Variable(image), Variable(label)

        pred = model(image)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%50 == 0:
            loss_float = float(loss.numpy())
            losses.append(loss_float)

            if i%1000 == 0:
                print("Current loss: {:.4f}".format(loss_float))

    logging.info("Saving state dict...")
    with open(path_params, "wb") as fh:
        torch.save({'state_dict': model.state_dict()}, fh)

    plt.plot(losses)
    plt.show()


def main():
    parser = ArgumentParser(description="Train and evaluate resnet model.")
    parser.add_argument("--params", help="Path to file to save params",
                        default="./data/cdiscount/model_params.torch")
    args = parser.parse_args()

    setup_logging()

    train(args.params)


if __name__ == "__main__":
    main()
