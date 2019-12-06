
import io
import torch 
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image 

import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Transfer Learning
        # leveraging the feature extractor of ResNet-152 
        #  #obtain the ResNet model from torchvision.model library
        self.model = models.resnet50(pretrained=True)

        # Building classifier and since we are classifying the images
        # into NORMAL and PNEMONIA, we output a two-dimensional tensor.
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fc.in_features,2),
            nn.LogSoftmax(dim=1)
        )

        # Pytorch provides us with the ability to take and freeze 
        # these powerful feature extractors, attach our own classifiers 
        # depending on our problem domain and train the resulting model to suit our problem
        # Requires_grad = False 
        # denies the ResNet model the ability 
        # to update its parameters hence make it unable to train.
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.fc = self.classifier

    # Every model built from the nn.Module requires 
    # that we override the  forward function    
    def forward(self, x):
        return self.model(x)
    

def get_model():
    checkpoint_path='classifier.pt'
    model=models.resnet50(pretrained=True)
    model.classifier = Model()
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'),strict=False)
    model.eval()
    return model

def get_tensor(image_bytes):
	my_transforms=transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
	image=Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)
        
