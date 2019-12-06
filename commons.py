
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
        self.model = torchvision.models.resnet50(pretrained=True)

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
    

    # Pytorch Blitz: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

    def fit(self, dataloaders, num_epochs):
        train_on_gpu = torch.cuda.is_available()
        optimizer = optim.Adam(self.model.fc.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, 4)
        criterion = nn.NLLLoss()
        since = time.time()
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc =0.0
        if train_on_gpu:
            self.model = self.model.cuda()
        for epoch in range(1, num_epochs+1):
            print("epoch {}/{}".format(epoch, num_epochs))
            print("-" * 10)
            
            for phase in ['train','test']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()
                
                running_loss = 0.0
                running_corrects = 0.0
                
                for inputs, labels in dataloaders[phase]:
                    if train_on_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print("{} loss:  {:.4f}  acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
                
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        
        time_elapsed = time.time() - since
        print('time completed: {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 600))
        print("best val acc: {:.4f}".format(best_acc))
        
        self.model.load_state_dict(best_model_wts)
        return self.model

def get_model():
    checkpoint_path='classifier.pt'
    model=models.resnet50(pretrained=True)
    model.classifier = classifier()
    model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'),strict=False)
    model.eval()
    return model

def get_tensor(image_bytes):
	my_transforms=transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
	image=Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)
        