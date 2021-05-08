from torch import nn
import torch
from itertools import repeat
# from dropblock import dropblock


class Spatial_Dropout(nn.Module):
    def __init__(self,drop_prob):

        super(Spatial_Dropout,self).__init__()
        self.drop_prob = drop_prob

    def forward(self,inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self,input):
        return input.new().resize_(input.size(0),*repeat(1, input.dim() - 2),input.size(2))

# class Spatial_Dropout(nn.Module):
#     def __init__(self, drop_prob):
#         super(Spatial_Dropout, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         assert x.ndim >= 3
#         if not self.training or self.drop_prob == 0:
#             return x
#
#         noise = self._make_noise(x)
#         x = torch.mul(x, noise)
#         return x
#
#     def _make_noise(self, x):
#         x1 = torch.dropout(torch.ones(x.shape[2:]), self.drop_prob, True)
#         x1 = x1.expand_as(x)
#         x1 = x1.to(x.device)
#         return x1

# class Dropout1d(nn.Module):
#     def __init__(self, p):
#         super(Dropout1d, self).__init__()
#         self.p = p
#
#
#     def forward(self, x):
#         x = x.unsqueeze(-1)
#         x = nn.Dropout2d(self.p)(x)
#         x = x.squeeze(-1)
#         return x




class ResBlock(nn.Module):
    def __init__(self, in_channel, filters, layer_shape, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, filters, 3, 1,padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, filters, 3, 1,padding=1),
            nn.ReLU(),
            # nn.LayerNorm(layer_shape),
            nn.BatchNorm2d(filters),

            nn.Conv2d(filters, filters, kernel_size, 1, kernel_size//2 ),
            nn.ReLU(),
            # nn.LayerNorm(layer_shape),
            nn.BatchNorm2d(filters),
            nn.Conv2d(filters, filters, 3, 1,padding=1),
            nn.ReLU(),
            # nn.LayerNorm(layer_shape)
            nn.BatchNorm2d(filters)
        )


    def forward(self, x):
        x1 = self.block(x)
        x = self.conv(x)
        x = x + x1
        return x





class ResBlock2(nn.Module):
    def __init__(self, in_channel=3, filters=128, kernel_size=3):
        super(ResBlock2, self).__init__()

        # self.block1 = ResBlock(in_channel, filters, [filters, 60], kernel_size)
        self.block1 = ResBlock(in_channel, filters, [160,filters, 224, 224], kernel_size)
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Spatial_Dropout(0.3),
            # Dropout1d(0.3)
            # nn.Dropout(0.3)
        )
        # self.block2 = ResBlock(filters, filters//2, [filters//2, 30], kernel_size)
        self.block2 = ResBlock(filters, filters // 2, [filters // 2, 1,1], kernel_size)
        self.pool2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # forward
        x = self.block1(x)
        x = self.pool2(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        return x

class ResNet_Tri(nn.Module):
    def __init__(self, in_channel=3, num_classes=19, filters=128):
        super(ResNet_Tri, self).__init__()
        self.seq_3 = ResBlock2(in_channel=in_channel, filters=filters, kernel_size=3)
        self.seq_5 = ResBlock2(in_channel=in_channel, filters=filters, kernel_size=5)
        self.seq_7 = ResBlock2(in_channel=in_channel, filters=filters, kernel_size=7)

        self.classifier = nn.Sequential(
            nn.Linear(192, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)

            # nn.Linear(192, 128),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(128, num_classes),
            # nn.Softmax()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # 输入为(1, 60, 8) 更改格式为(8, 60)
        # x = x.permute(0, 3, 2, 1).squeeze(-1)
        # x = x.permute(0, 3, 2, 1)
        # print(x.shape)
        x1 = self.seq_3(x)
        x2 = self.seq_5(x)
        x3 = self.seq_7(x)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=1, filters=128):
        super(ResNet, self).__init__()
        self.seq = ResBlock2(in_channel=in_channel, filters=filters, kernel_size=3)

        self.classifier = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)

        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # 输入为(1, 60, 8) 更改格式为(8, 60)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        # print(x.shape)
        x = self.seq(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x






if __name__ == "__main__":
    # from torchsummary.torchsummary import summary
    model = ResNet_Tri(in_channel=3, filters=128, num_classes=1)
    # summary(model.cuda(), input_size=(8, 60))
    # summary(model.cuda(), input_size=(1, 60, 8))

    x = torch.rand([8, 3,224, 224 ])
    y = model(x)
    print(model.classifier)

