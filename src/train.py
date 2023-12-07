import numpy as np
import sys
from util import load_data
from models import ResNetCallig, VGG16, Model
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop, Adagrad
from pprint import pprint
import threading

some_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

my_adam = lambda *ar, **kw: Adam(*ar, **kw, amsgrad=False)
my_sgd = lambda *ar, **kw: SGD(*ar, **kw, nesterov=True, momentum=0.9)
my_adam.__name__ = 'AdamNoAMS'; my_sgd.__name__ = 'SGDNest0.9'

param_grid = {
    'optimizer' : [my_adam, my_sgd], #RMSprop, Adagrad],
    'lr' : np.logspace(-5, -1, 3),
    'weight_decay' : [0.]# np.logspace(-5, -2, 4),
}

def grid_search(model_name, dataset, param_grid):
    model = Model(model_name, dataset)
    m = len(param_grid['optimizer'])
    n = len(param_grid['lr'])
    k = len(param_grid['weight_decay'])
    final_losses = np.zeros((m,n,k))
    param_descrip = []
    for i, opt in enumerate(param_grid['optimizer']):
        for j, lr in enumerate(param_grid['lr']):
            for k, wd in enumerate(param_grid['weight_decay']):
                hist = model.train(
                    num_epochs=7,
                    alpha=lr,
                    optimizer=opt,
                    weight_decay=wd
                )
                final_losses[i,j,k] = hist[-1]['val_acc']
                param_descrip.append((opt.__name__, str(lr), str(wd)))

    pprint(final_losses)
    pprint(param_descrip)
    max_ind = np.argmax(final_losses)
    print(param_descrip[max_ind])

    return final_losses, param_descrip

best_params = {
    'optimizer': my_adam,
    'alpha': 0.0001,
    'weight_decay': 0.001,
}

best_params_vgg = {
    'optimizer': my_adam,
    'alpha': 0.001,
    #'weight_decay': 0.001,
}

def train_and_save(model, num_epochs, params):
    model.train(num_epochs, **params)
    model.eval_model()
    model.save_model()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        num_class = int(sys.argv[1])
        datasets = load_data([num_class], transform = some_transforms, batch_size=256, refresh=True)
        #model = Model(ResNetCallig, datasets[0])
        model = Model(VGG16, datasets[0])
        train_and_save(model, 6, best_params_vgg)
    else:
        datasets = load_data(transform = some_transforms, batch_size=256, refresh=True)
        #losses, descrips = grid_search(ResNetCallig, datasets[0], param_grid)

        vgg_models = [Model(VGG16, data) for data in datasets]
        resnet_models = [Model(ResNetCallig, data) for data in datasets]
        
        for model in resnet_models:
            train_and_save(model, 100, best_params)
