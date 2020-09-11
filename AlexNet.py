import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision import transforms

# AlexNet網絡結構
class MineAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MineAlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # softmax
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.logsoftmax(x)
        return x



# 讀取提前下載的預訓練模型
state_dict = torch.load(r'E:\google drive\iii_course\HOT\img_distinct\alexnet-owt-4df8aa71.pth')
alexNet = MineAlexNet(1000)
alexNet.load_state_dict(state_dict)

# 修改AlexNet網絡結構的後兩層
alexNet.classifier[6] = nn.Linear(4096, 2)

# 保存包含了預訓練參數的，貓狗分類AlexNet模型
torch.save(alexNet.state_dict(), 'begin.pth')

import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision import transforms

# 數據預處理
# 非常重要，如果不處理，效果可能非常差！！！
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(size=(227,227)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #將圖片轉換爲Tensor,歸一化至[0,1]
    normalize
])

# 從文件夾中讀取訓練數據
train_dataset = ImageFolder(r'E:\google drive\iii_course\HOT\img_distinct\classify\test_classify',transform=transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

# 從文件夾中讀取validation數據
validation_dataset = ImageFolder(r'E:\google drive\iii_course\HOT\img_distinct\classify\test_classify',transform=transform)
validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=True)

# AlexNet
class MineAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MineAlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.logsoftmax(x)
        return x


# 讀取轉換後的AlexNet模型
state_dict = torch.load('begin.pth')
alexNet = MineAlexNet(2)
alexNet.load_state_dict(state_dict)

# cuda
alexNet.cuda()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexNet.parameters(), lr=0.00005)

epochs = 30
train_losses, validation_losses = [], []

# 訓練
for e in range(epochs):
    running_loss = 0
    for images,labels in trainloader:

        images = images.cuda()
        labels = labels.cuda()

        # TODO: Training pass
        optimizer.zero_grad()

        output = alexNet(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        validation_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in validationloader:
                images = images.cuda()
                labels = labels.cuda()

                log_ps = alexNet(images)
                validation_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainloader))
        validation_losses.append(validation_loss/len(validationloader))

        torch.save(alexNet.state_dict(), str(e+1+37) +'.pth')

        print("Epoch: {}/{}.. ".format( e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(validationloader)))

# 畫一下精度圖
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')
plt.plot(validation_losses, label='Validation loss')
plt.legend(frameon=False)