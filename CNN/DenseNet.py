# dense_block_layer within BottleNeck
class dense_block_layer(nn.Module):
    def __init__(self, input_channel, growth_rate):
        super(dense_block_layer, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel, 4 * growth_rate, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.cat([out, x], 1)
        return out

    
class Transition(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(Transition, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)
        out = F.avg_pool2d(out, 2)
        return out
    
class DenseNet(nn.Module):
    def __init__(self, LayerNums, growth_rate, compression_rate, num_classes):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate

        in_channels = 2 * growth_rate

        self.conv = nn.Sequential(
            nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(2 * growth_rate),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.dense1 = self.make_dense_block(in_channels, LayerNums[0])
        in_channels += LayerNums[0] * growth_rate
        out_channels = int(math.floor(in_channels * compression_rate))
        self.trans1 = Transition(in_channels, out_channels)
        in_channels = out_channels

        self.dense2 = self.make_dense_block(in_channels, LayerNums[1])
        in_channels += LayerNums[1] * growth_rate
        out_channels = int(math.floor(in_channels * compression_rate))
        self.trans2 = Transition(in_channels, out_channels)
        in_channels = out_channels

        self.dense3 = self.make_dense_block(in_channels, LayerNums[2])
        in_channels += LayerNums[2] * growth_rate

        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(128, 10)
        )

    def make_dense_block(self, in_channels, LayersNum):
        layers = []
        for i in range(LayersNum):
            layers.append(dense_block_layer(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.AdaptiveAvgPool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

