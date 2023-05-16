# SE Block
class SE(nn.Module):
    def __init__(self, in_channels, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels, in_channels//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channels//ratio, in_channels, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)


# SE ResBlock
class SE_ResBlock(nn.Module):
    def __init__(self, channel, ratio):
        super(SE_ResBlock, self).__init__()
        self.channel = channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel,
                      out_channels=channel,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel)
        )
        self.se = SE(channel, ratio)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        coefficient = self.se(out)
        out = out * coefficient
        out += x
        out = F.relu(out)
        return out

    
# SE ResNet
class SE_ResNet(nn.Module):
    def __init__(self, ratio):
        super(SE_ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.resblock1 = SE_ResBlock(32, ratio)
        self.resblock2 = SE_ResBlock(16, ratio)
        self.fc = nn.Linear(400, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.conv2(out)
        out = self.resblock2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out





    

net = SE_ResNet(ratio=16)
print(net)