import torch
import torch.nn as nn
import torch.nn.functional as F

import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self, num_classes):
        super().__init__()

        self.fc8 = nn.Conv2d(4096, 21, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]

    def forward(self, x):
        d = super().forward_as_dict(x)
        x = super().forward(x)

        cam = self.fc8(x)

        _, _, h, w = cam.size()
        pred = F.avg_pool2d(cam, kernel_size=(h, w), padding=0)

        pred = pred.view(pred.size(0), -1)
        conv4 = d['conv4'].detach()
        conv5 = d['conv5'].detach()
        conv6 = d['conv6'].detach()
        return cam, pred, [conv4, conv5, conv6]

    def forward_cam(self, x):
        x = super().forward(x)
        cam = self.fc8(x)

        return cam

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
