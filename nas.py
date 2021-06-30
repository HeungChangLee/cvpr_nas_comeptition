import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from RandAugment import RandAugment
import torchvision
from torchvision.transforms import transforms
from shakeshake_models import ShakeResNet, ShakeResNeXt

class RandAugResnet(nn.Module):
    def __init__(self, model, transform, min_values, scale_values, input_size):
        super(RandAugResnet, self).__init__()
        self.model = model
        self.transform = transform
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.eval_transform = transforms.Compose([
            transforms.Normalize(min_values, scale_values),
            transforms.ToPILImage(),
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()
        ]) #32x32
        
    def forward(self, x):
        if self.training:
            xs = []
            for x_ in x:
                x_ = self.transform(x_)
                xs.append(x_)
            xs = torch.stack(xs)
            x = self.model(xs.to(self.device))
        else:
            xs = []
            for x_ in x:
                x_ = self.eval_transform(x_)
                xs.append(x_)
            xs = torch.stack(xs)
            x = self.model(xs.to(self.device))
        return x

class NAS:
    def __init__(self):
        pass

    def search(self, train_x, train_y, valid_x, valid_y, metadata):
        n_classes = metadata['n_classes']
        
        channels = train_x.shape[1]
        min_values = []
        scale_values = []
        if train_x.shape[2] > train_x.shape[3]:
            input_size = train_x.shape[2] + 4
        else:
            input_size = train_x.shape[3] + 4
        for i in range(channels):
            min_values.append(np.min(train_x[:,i,:,:]))
            scale_values.append((np.max(train_x[:,i,:,:])-min_values[-1]))

        transform = transforms.Compose([
            transforms.Normalize(min_values, scale_values),
            transforms.ToPILImage(),
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()
        ]) #32x32
    
        # Loading model
        if input_size < 40:
            model = ShakeResNet(32, 64, n_classes)
            model.c_in = nn.Conv2d(train_x.shape[1], model.in_chs[0], 3, padding=1)      
            transform.transforms.insert(3, RandAugment(2, 3))
            model = RandAugResnet(model, transform, min_values, scale_values, input_size)
        else:
            model = torchvision.models.densenet121()
            model.features[0] = nn.Conv2d(train_x.shape[1], 64, kernel_size=7, stride=1, padding=3, bias=False)
            model.classifier = nn.Linear(1024, n_classes)

        return model
