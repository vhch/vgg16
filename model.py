import torch
import torch.nn as nn
#VGG16
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            # 첫번째층
            # ImgIn shape=(batch_size, 3, 224, 224)
            #    Conv     -> (batch_size, 64, 224, 224)
            #    Pool     -> (batch_size, 64, 112, 112)
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 두번째층
            # ImgIn shape=(batch_size, 64, 112, 112)
            #    Conv     -> (batch_size, 128, 112, 112)
            #    Pool     -> (batch_size, 128, 56, 56)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 세번째층
            # ImgIn shape=(batch_size, 128, 56, 56)
            #    Conv     -> (batch_size, 256, 28, 28)
            #    Pool     -> (batch_size, 256, 28, 28)
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 네번째층
            # ImgIn shape=(batch_size, 256, 28, 28)
            #    Conv     -> (batch_size, 512, 14, 14)
            #    Pool     -> (batch_size, 512, 14, 14)
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # 다섯번째층
            # ImgIn shape=(batch_size, 512, 14, 14)
            #    Conv     -> (batch_size, 512, 14, 14)
            #    Pool     -> (batch_size, 512, 7, 7)
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            # 여섯번째층
            # ImgIn shape=(batch_size, 512, 7, 7)
            #    reshape     -> (batch_size, 512 * 7 * 7)
            #    Linear     -> (batch_size, 4096)
            nn.Linear(25088, 4096), nn.ReLU(True), nn.Dropout(),
            # 일곱번째층
            # ImgIn shape=(batch_size, 4096)
            #    Linear     -> (batch_size, 4096)
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            # 여덟번째층
            # ImgIn shape=(batch_size, 4096)
            #    Linear     -> (batch_size, 1000)
            nn.Linear(4096, 1000)
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x