from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


# Adjust the model to get a higher performance
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)    # 28*28 →  (28+1-3)  26*26
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, 64)  #full connection layer
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # x: 1*28*28
        x = self.conv1(x)  # 26*26*8
        x = F.relu(x)
        x = self.conv2(x)   # 24*24*16
        x = F.relu(x)
        x = F.max_pool2d(x, 2)   # 12*12*16 = 2304
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# 训练
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    plt.figure()
    pic = None
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx in (1,2,3,4,5):
            if batch_idx == 1:
                pic = data[0,0,:,:]
            else:
                pic = torch.cat((pic,data[0,0,:,:]),dim=1)
        data, target = data.to(device), target.to(device)
        # SGD
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # Calculate gradients
        loss.backward()
        # Optimize the parameters according to the calculated gradients
        optimizer.step()


        if batch_idx % args.log_interval == 0:  # 10
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(   #轮次
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    # plt.imshow(pic.cpu(), cmap='gray')
    # plt.show()



# 测试
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # 这样就不会计算grad
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # batch loss                                   加起来
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss

            # prediction         维度=1
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # 看跟target有多少是相同的
            correct += pred.eq(target.view_as(pred)).sum().item()

            result = pred.eq(target.view_as(pred))

        fig, axes = plt.subplots(3, 3, constrained_layout=True)  # Misclassified images
        number = 0
        for i in range(len(result)):
            if (number == 9):
                break
            if (result[i] == False):
                axes[int(number / 3), number % 3].imshow(data[i, 0, :, :].cpu(), cmap='gray')
                x = np.round(output[i].cpu().numpy() / (output[i].cpu().numpy().sum() + 1e-5), 2)
                axes[int(number / 3), number % 3].set_title("lable=" + str(target[i].cpu().numpy()) + ",predict=" + str(
                    pred[i].cpu().numpy()) + "\nprobability:\n" + str(x), fontsize=6)
                axes[int(number / 3), number % 3].set_xticks([])
                axes[int(number / 3), number % 3].set_yticks([])
                number = number + 1
        plt.show()

        fig, axes = plt.subplots(3, 3, constrained_layout=True)  # Well classified images
        number = 0
        for i in range(len(result)):
            if (number == 9):
                break
            if (result[i] == True):
                axes[int(number / 3), number % 3].imshow(data[i, 0, :, :].cpu(), cmap='gray')
                x = np.round(output[i].cpu().numpy() / (output[i].cpu().numpy().sum() + 1e-5), 2)
                axes[int(number / 3), number % 3].set_title("lable=" + str(target[i].cpu().numpy()) + ",predict=" + str(
                    pred[i].cpu().numpy()) + "\nprobability:\n" + str(x), fontsize=6)
                axes[int(number / 3), number % 3].set_xticks([])
                axes[int(number / 3), number % 3].set_yticks([])
                number = number + 1
        plt.show()

    # compute loss
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        # accuracy percentage
        100. * correct / len(test_loader.dataset)))






def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # batch_size = 64
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # test_batch_size = 1000
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    # epoch
    parser.add_argument('--epochs', type=int, default=14, metavar='N',     #循环多少次
                        help='number of epochs to train (default: 14)')
    # learning rate = 1.0
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    # gamma value
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    # seed = 1
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # log_interval = 10
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # save_model
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print (torch.cuda.is_available())   #检测一下是否用到CUDA

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    # 判断是用GPU还是CPU训练
    print (device)

    # batch_size is a crucial hyper-parameter
    # (超参数，用来定义模型结构或优化策略) 批处理，每次处理的数据数量
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        # Adjust num worker and pin memory according to your computer performance
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalize the input (black and white image)
    # 变换，转换为tensor张量
    # data = [d[0].data.cpu().numpy()for d in mnist_data]
    # np.mean(data)
    transform=transforms.Compose([
    #also can use: pipeline = transforms.Compose([
        transforms.ToTensor(), # convert images into "tensor"
        # 正则化（when overfitting，降低模型复杂度）
        # mean均值 : 0.13066062,  std标准差: 0.30810776
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Make train dataset split
    # download the MNIST dataset automatically 下载数据集
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    # Make test dataset split
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    # Convert the dataset to dataloader, including train_kwargs and test_kwargs
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Put the model on the GPU or CPU
    model = Net().to(device)

    # Create optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Create a schedule for the optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Begin training and testing
    # 跑
    for epoch in range(1, args.epochs + 1):
        # 训练
        train(args, model, device, train_loader, optimizer, epoch)
        # 测试
        test(model, device, test_loader)
        scheduler.step()

    # Save the model 保存模型
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")







if __name__ == '__main__':
    main()
