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


def make_model(state_dict=None, volatile=False):
    model = torchvision.models.resnet34(num_classes=5270)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if volatile:
        model.conv1.volatile = True
    model = model.cuda()
    return model


def train(path_params):
    """
    Train the resnet from scratch using cdiscount data
    :param path_params: The path where to save the params
    """
    dataset = CDiscountDataset('data/cdiscount', train=True, transform=data_transforms['train'])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    model = make_model()
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train a single epoch

    losses = []
    i = 0
    try:
        for i, data in enumerate(dataloader, 1):

            image, label = data
            image, label = Variable(image.cuda()), Variable(label.cuda())

            pred = model(image)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%25 == 0:
                loss_float = float(loss.data[0])
                losses.append(loss_float)

                if i%500 == 0:
                    print("Current loss: {:.4f}".format(loss_float))

    except KeyboardInterrupt:
        logging.info("Interrupted prematurely at iteration {}".format(i))

    logging.info("Saving state dict...")
    with open(path_params, "wb") as fh:
        torch.save({'state_dict': model.state_dict()}, fh)

    plt.plot(losses)
    plt.show()


def test(path_params):
    """
    Test the model on a pretrained net
    :param path_params: The path to the saved weights and biases
    :return: The accuracy on the validation set
    """
    dataset = CDiscountDataset('data/cdiscount', train=False, transform=data_transforms['val'])
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8)

    with open(path_params, "rb") as fh:
        state_dict = torch.load(fh)['state_dict']
    model = make_model(state_dict, volatile=True)  # volatile is required

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())  # for some reason, this is also required to prevent memory blow up?!

    # todo - why does memory blow up with certain settings?

    i = 0
    num_correct = 0
    try:
        for i, data in enumerate(dataloader, 1):

            image, label = data
            image, label = Variable(image.cuda()), Variable(label.cuda())

            output = model(image)
            loss = criterion(output, label)

            _, preds = torch.max(output.data, 1)
            num_correct += torch.sum(preds == label.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        num_processed = len(dataset)

    except KeyboardInterrupt:
        logging.info("Interrupted prematurely at iteration {}".format(i))
        num_processed = i * batch_size

    accuracy = num_correct / num_processed

    return accuracy


def main():
    parser = ArgumentParser(description="Train and evaluate resnet model.")
    parser.add_argument("--params", help="Path to file to save params",
                        default="./data/cdiscount/model_params.torch")
    parser.add_argument("--train", action='store_true',
                        help="Train model? Otherwise, will test")

    args = parser.parse_args()

    setup_logging()

    if args.train:
        train(args.params)
    else:
        accuracy = test(args.params)
        print("Accuracy: {:.2f}%".format(accuracy * 100))


if __name__ == "__main__":
    main()
