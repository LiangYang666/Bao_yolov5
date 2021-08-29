import types

import torch.nn as nn
import torch.nn.functional as F
import torch
import ipdb
import torchvision
from torch.autograd import Variable

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def resnet18_embedding():
    model = torchvision.models.resnet18(pretrained=True)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    model._forward_impl = types.MethodType(_forward_impl, model)
    del model.fc
    return model
# ckp_dir = '/media/D_4TB/YL_4TB/Competitions/ZNLSG_21_XinYe/data/cssjj/rundata/metalearning/models_trained/2ResNet50_dataaug_fc512_l2/checkpoints/checkpoint_30.pt'
# ckp_dir = '/media/D_4TB/YL_4TB/Competitions/ZNLSG_21_XinYe/data/cssjj/rundata/metalearning/models_trained/3ResNet50_dataaug_fc512_cos/checkpoints/checkpoint_315.pt'
ckp_dir = '/media/D_4TB/YL_4TB/Competitions/ZNLSG_21_XinYe/data/cssjj/rundata/metalearning/models_trained/3ResNet50_w100_dataaug_fc512_cos/checkpoints/checkpoint_195.pt'

def prototypical_net(pretrained_dict = ckp_dir):
    model = torchvision.models.__dict__['resnet50'](pretrained=False)
    # model.fc = Identity()
    lin = nn.Linear(model.fc.in_features, 512)
    model.fc = lin
    print('Loadding model from\n'
          '\t'+ckp_dir)
    checkpoint = torch.load(pretrained_dict)
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ =="__main__":

    input = torch.randn(3, 1, 608, 608) # input.is_leaf -> True,input.requires_grad -> False

    # torchvision.models.resnext50_32x4d()
    model = resnet18_embedding()
    # model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3,
    #                            bias=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    input = torch.randn(3, 3, 224, 224)
    # model.eval()
    o = model(input)
    print(o.shape)
    torch.save({'img_name': o.cpu()}, 'out.pt')
    out = torch.load('out.pt')
    # model.train()
    # for key in model.state_dict().keys():
        # print(key, ' ', model.state_dict()[key].is_leaf)
        # print(key, ' ', model.state_dict()[key].requires_grad)

