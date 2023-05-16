# ResBlock
class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
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
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        out = F.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
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
        self.resblock1 = ResBlock(32)
        self.resblock2 = ResBlock(16)
        self.fc = nn.Linear(400, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.conv2(out)
        out = self.resblock2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out