import time
import cv2
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn.init as init

from util import device, to_device

#print('Using device:', device)

# from https://www.kaggle.com/code/kauvinlucas/calligraphy-style-classification
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result, elapsed_time):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, time: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc'], elapsed_time))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

THREE_HOURS = 3*60*60
CUTOFF_TIME = THREE_HOURS

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD,weight_decay=1e-4,last_epoch=None):
    history = []
    optimizer = opt_func(model.parameters(), lr=lr, weight_decay=weight_decay)
    if last_epoch:
        epoch_range = range(last_epoch+1, last_epoch+1+epochs)
    else:
        epoch_range = range(epochs)
    overall_start = time.time()
    for epoch in epoch_range:
        epoch_start = time.time()
        # Training Phase 
        model.train()
        train_losses = []
        for i, batch in enumerate(train_loader):
            #if i > len(train_loader)//100: break
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result, time.time()-epoch_start)
        history.append(result)

        # check termination condition
        if (result['val_acc'] >= 0.96):# or (time.time() - overall_start) > THREE_HOURS):
            break

    return history


class VGG16(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes))
        self.alpha = 0.001
        self.name = 'VGG16'
        
    def forward(self, xb):
        return self.network(xb)


class ResNetCallig(ResNet, ImageClassificationBase):
    def __init__(self, num_classes):
        ImageClassificationBase.__init__(self)
        ResNet.__init__(self, BasicBlock, [2,2,2,2], num_classes=num_classes)
        self.alpha = 0.0005
        self.name = 'ResNet'

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
        
class Model:
    def __init__(self, model_name, dataset):
        self.data = dataset
        self.model = model_name(dataset.num_classes)
        self.model_path = f'saved_models/{self.model.name}_{self.data.num_classes}.pth'
        
        self.model = to_device(self.model, device);
    
    def train(self, num_epochs=100, alpha=None, optimizer=torch.optim.SGD, weight_decay=1e-4):
        if not alpha:
            alpha = self.model.alpha
        print('\nNow Training', self.data.num_classes, 'with',
              f'{num_epochs=}, {alpha=}, {optimizer.__name__}, {weight_decay=}')
        history = fit(num_epochs, alpha, self.model,
                      self.data.nn_train_dl, self.data.nn_eval_dl, optimizer, weight_decay)
        return history
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def predict_image(self, img, model):
        # Convert to a batch of 1
        xb = img.unsqueeze(0)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        return self.test.classes[preds[0].item()]

    def plot_prediction(test_ind, model):
        img, label = self.test[test_ind]
        label = self.test.classes[label]
        predicted = predict_image(self.test, img, model)
        plt.imshow(img.permute((1,2,0)))
        plt.title(f'{label=}, {predicted=}')

    def eval_model(self):
        res = evaluate(self.model, self.data.test_dl)
        print(res)

    def get_intermediate_activation(self, img, skip_final=8):
        is_numpy = False
        if type(img) != torch.Tensor:
            is_numpy = True
            n = round(np.sqrt(len(img)))
            img = img.reshape(n,n)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = torch.tensor(img).permute(2,0,1)
        x = img.unsqueeze(0)
        if type(self.model) == VGG16:
            for layer in self.model.network[:-skip_final]:
                x = layer(x)
        else:
            for layer in list(self.model.children())[:-skip_final]:
                x = layer(x)
        if is_numpy:
            x = x.detach().numpy()
            if len(x.shape) > 2:
                x = np.mean(x, axis=(2,3))
            x = x.reshape(-1)
        return x
