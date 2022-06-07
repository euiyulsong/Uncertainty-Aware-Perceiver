import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse
from models import resnextto
from models import lenet, vgg, resnet, resnet_v2, resnext, googlenet, densenet
from utils import progress_bar, adjust_learning_rate, model_log

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="lenet", type=str, help='choose which model to train')
parser.add_argument('--dataset', default="CIFAR100", type=str, help='chosse CIFAR10 or CIFAR100')
parser.add_argument('--pretrain', default=False, type=bool, help="use pretrain model?")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else 'cpu'
best_acc = 0

print("==> Preparing data......")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# data set
from torch.utils.data import random_split

if args.dataset == "CIFAR10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_size = len(trainset) - int(0.1 * len(trainset))
    train_ds, validset = random_split(trainset, [train_size, int(0.1 * len(trainset))])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 10
elif args.dataset == "CIFAR100":
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_size = len(trainset) - int(0.1 * len(trainset))
    train_ds, validset = random_split(trainset, [train_size, int(0.1 * len(trainset))])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    num_classes = 100
else:
    raise "only support dataset CIFAR10 or CIFAR100"

if args.model == "lenet":
    net = lenet.LeNet(num_classes=num_classes)

elif args.model == "vgg16":
    net = vgg.vgg16(num_classes=num_classes, pretrained=args.pretrain)
elif args.model == "vgg16_bn":
    net = vgg.vgg16_bn(num_classes=num_classes, pretrained=args.pretrain)

elif args.model == "resnet18":
    net = resnet.resnet18(num_classes=num_classes, pretrained=args.pretrain)
elif args.model == "resnet34":
    net = resnet.resnet18(num_classes=num_classes, pretrained=args.pretrain)
elif args.model == "resnet50":
    net = resnet.resnet50(num_classes=num_classes, pretrained=args.pretrain)

elif args.model == "resnetv2_18":
    net = resnet_v2.resnet18(num_classes=num_classes, pretrained=args.pretrain)
elif args.model == "resnetv2_34":
    net = resnet_v2.resnet18(num_classes=num_classes, pretrained=args.pretrain)
elif args.model == "resnetv2_50":
    net = resnet_v2.resnet50(num_classes=num_classes, pretrained=args.pretrain)

elif args.model == "resnext50_32x4d":
    net = resnextto.resnext50_32x4d(num_classes=num_classes, pretrained=args.pretrain)
elif args.model == "resnext101_32x8d":
    net = resnextto.resnext101_32x8d(num_classes=num_classes, pretrained=args.pretrain)


elif args.model == "resnext50":
    net = resnext.resnext50(num_classes=num_classes)
elif args.model == "googlenet":
    net = googlenet.GoogLeNet(num_classes=num_classes)
elif args.model == "densenet":
    net = densenet.DenseNet_CIFAR(num_classes=num_classes)
else:
    raise "please check model"

# freeze
# count = 0
# for param in net.parameters():
#     count += 1
# for i, param in enumerate(net.parameters()):
#     if i <= count-1 - 10:
#         param.requires_grad = False

str_pretrain = ""
if args.pretrain:
    str_pretrain = "pretrain_"

model_name = args.model + "_" + str_pretrain + args.dataset + ".pth"
log_name = args.model + "_" + str_pretrain + args.dataset + ".log"

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)


def train(epoch=None):
    max_loss = float("inf")
    count = 5
    for epoch in range(150):

        global net, optimizer
        optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)
        print('Epoch {0}'.format(epoch))
        net.train()
        train_loss = 0.0
        correct_count = 0
        total_num = 0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_num += targets.size(0)
            correct_count += predicted.eq(targets).sum().item()

        print(i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(i+1), 100.*correct_count/total_num, correct_count, total_num))

        # test
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total

        if test_loss/(batch_idx+1) < max_loss:
            max_loss = test_loss/(batch_idx+1)
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if not os.path.isdir('log'):
                os.mkdir('log')
            torch.save(net.state_dict(), os.path.join('./checkpoint/', model_name))
            best_acc = acc
            model_log(model_name, str(best_acc), os.path.join('./log/', log_name))
            count -= 1
        else:
            count = 5
        if count == 0:
            break
    net.load_state_dict(torch.load(f'./checkpoint/{model_name}'))

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
if __name__ == "__main__":
    train()