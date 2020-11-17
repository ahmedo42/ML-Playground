import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import math
import argparse
from GoogLeNet import GoogLeNet


parser = argparse.ArgumentParser(
    description="configure training hyperparameters")
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--aux', type=bool, default=False,
                    help='auxillary classifiers')


args = parser.parse_args()
writer = SummaryWriter()


def validate(val_loader, model, criterion):

    model.eval()
    total = correct = loss = 0
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs,aux1,aux2 = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
        loss += criterion(outputs,labels)
        if args.aux:
            loss += criterion(aux1,labels) + criterion(aux2,labels)

    acc = 100 * correct/total
    return loss , acc


def train(train_loader, model,criterion,optimizer,scheduler,epochs):
    best_acc = 0
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            output,aux1,aux2 = model(images)
            loss = criterion(output, labels)
            if args.aux:
                aux1_loss = 0.3*criterion(aux1,labels) 
                aux2_loss = 0.3*criterion(aux2,labels) 
                aux1_loss.backward()
                aux2_loss.backward()
            loss.backward()
            optimizer.step()
        val_loss , val_acc = validate(test_loader, model, criterion)
        train_loss , train_acc = validate(train_loader,model,criterion)
        writer.add_scalar('Loss/Train',train_loss,epoch)
        writer.add_scalar('Accuracy/Train',train_acc,epoch)
        writer.add_scalar('Loss/Val',val_loss,epoch)
        writer.add_scalar('Accuracy/Val',val_acc,epoch)
        if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'googlenet_cifar10.pkl')
        print("Epoch: {} ,validation acc: {} ,best acc: {}".format(epoch,val_acc,best_acc))
        scheduler.step()


if __name__ == "__main__":
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_data = torchvision.datasets.CIFAR10(
        '../data/CIFAR10/train', transform=train_transform, train=True, download=True)
    test_data = torchvision.datasets.CIFAR10(
        '../data/CIFAR10/test', transform=test_transform, train=False, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = GoogLeNet(n_classes=10,aux_logits=args.aux)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    if torch.cuda.is_available:
      torch.cuda.set_device(0)
    model.cuda()
    train(train_loader,model,criterion,optimizer,scheduler,EPOCHS)
    writer.flush()
    print("Compeleted Training")