import torch
import torchvision.models.vgg as vgg
from collections import namedtuple


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        try:
            from torchvision.models import VGG19_Weights
            self.vgg_layers = vgg.vgg19(weights=VGG19_Weights.DEFAULT).features
        except ImportError:
            self.vgg_layers = vgg.vgg19(pretrained=True).features

        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        '''
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        '''

        self.layer_name_mapping = {'3': "relu1", '8': "relu2"}

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
            if name == '8':
                break
        LossOutput = namedtuple("LossOutput", ["relu1", "relu2"])
        return LossOutput(**output)


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.model = LossNetwork()
        self.model.cuda()
        self.model.eval()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, x, target):
        x_feature = self.model(x[:, 0:3, :, :])
        target_feature = self.model(target[:, 0:3, :, :])

        feature_loss = (
            self.l1_loss(x_feature.relu1, target_feature.relu1) +
            self.l1_loss(x_feature.relu2, target_feature.relu2)) / 2.0

        l1_loss = self.l1_loss(x, target)
        l2_loss = self.mse_loss(x, target)

        loss = feature_loss + l1_loss + l2_loss

        return loss
