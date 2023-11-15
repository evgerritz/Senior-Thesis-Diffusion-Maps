import os
import cv2
import numpy as np
import pickle

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from PIL import ImageFilter

def get_default_device(override=False):
    """Pick GPU if available, else CPU"""
    gpu = 0
    use_cuda = torch.cuda.is_available() and not override 
    device = torch.device(gpu if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(gpu)
    return device
   
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
class GaussianBlur(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))  
    
class RandomNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img)
        noise = np.random.normal(self.mean, self.std, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
        return Image.fromarray(noisy_img)

class Dataset:
    def __init__(self, train_dir_name, test_dir_name, transform=None, batch_size=128):
        if not transform:
            transform = ToTensor()
        self.data_dir = '../data/'
        self.train = ImageFolder(self.data_dir + train_dir_name, transform=transform)
        self.test = ImageFolder(self.data_dir + test_dir_name, transform=transform)
        self.batch_size = batch_size
        self.classes = os.listdir(self.data_dir + train_dir_name)
        self.num_classes = len(self.classes)
        self.path = f'saved_objs/Dataset/{self.num_classes}.pkl'
        tX, ty = list(zip(*self.train))
        
        X, y = np.array(tX), np.array(ty)
        X = X.transpose((0,2,3,1))
        X = np.array([cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) for rgb in X])
        y = y.reshape(-1,1)
        
        self.X = X
        self.y = y
        
        # configure training dataloader
        random_seed = 1
        torch.manual_seed(random_seed);
        val_size = 10000
        train_size = len(self.train) - val_size
        nn_train, nn_eval = random_split(self.train, [train_size, val_size])

        nn_train_dl = DataLoader(nn_train, self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        nn_eval_dl = DataLoader(nn_eval, self.batch_size*2, num_workers=8, pin_memory=True)
        test_dl = DataLoader(self.test, self.batch_size*2, num_workers=8, pin_memory=True)
        self.nn_train_dl = DeviceDataLoader(nn_train_dl, device)
        self.nn_eval_dl = DeviceDataLoader(nn_eval_dl, device)
        self.test_dl = DeviceDataLoader(test_dl, device)


    def get_label(self, name=None, class_no=None):
        if name:
            return self.train.classes.index(name)
            
        if class_no:
            return self.train.classes[class_no]
        
    def take_subset(calligraphers, n_samples_each):
        # number of calligraphers
        c = len(calligraphers)
        
        subset_class_nos = [self.get_label(name=calligrapher) for calligrapher in calligraphers]
        
        dimx, dimy = self.X.shape[1:]

        subset_locs_mat = (self.y == subset_class_nos)

        X_reduced = np.stack([self.X[subset_locs_mat[:, i]][:n_samples_each] for i in range(c)])
        X_reduced = X_reduced.reshape(c*n_samples_each, dimy, dimx)
        y_reduced = np.repeat(subset_class_nos, n_samples_each).reshape(-1,1)

        return X_reduced, y_reduced
        
    @np.vectorize
    def full_label(self, label, chinese=True):
        label_to_full = {
            'wxz' : ('王羲之', 'Wang Xizhi'),
            'yzq' : ('颜真卿', 'Yan Zhenqing'),
            'lgq' : ('柳公权', 'Liu Gongquan'),
            'sgt' : ('孙过庭', 'Sun Guoting'),
            'smh' : ('沙孟海', 'Sha Menghai'),
            'mf'  : ('米芾', 'Mi Fu'),
            'htj'  : ('黄庭坚', 'Huang Tingjian'),
            'oyx' : ('欧阳询', 'Ouyang Xun'),
            'zmf' : ('赵孟頫', 'Zhao Mengfu'),
            'csl' : ('褚遂良', 'Chu Suiliang'),
            'wzm' : ('文征明', 'Wen Zhengming'),
            'lqs' : ('梁秋生', 'Liang Qiusheng'),
            'yyr' : ('于右任', 'Yu Youren'),
            'hy'  : ('弘一', 'Hong Yi'),
            'bdsr': ('八大山人', 'Bada Shanren'),
            'fwq' : ('范文强', 'Fan Wenqiang'),
            'gj'  : ('管峻', 'Guan Jun'),
            'shz' : ('宋徽宗', 'Song Huizong'),
            'mzd' : ('毛泽东', 'Mao Zedong'),
            'lx'  : ('鲁迅', 'Lu Xun'),
        }
        
        if chinese:
            return label_to_full[label][0]
        else:
            return label_to_full[label][1]

    def show_img(index, axis=None):
        if axis == None:
            _, axis = plt.subplots(1,1)
        axis.imshow(self.X[index], cmap='gray')
        axis.set_title(get_label(class_no = self.y[index][0]))

    def show_batch(self):
        for images, labels in self.nn_train_dl:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
            break
    
    def prep_for_embed(self, subset, num_samples):
        X_subset, y_subset = self.take_subset(subset, num_samples)
        X_sub_lin = X_subset.reshape(num_samples*len(subset), -1)
        return X_sub_lin, y_subset
    
    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self, f)
            
def load_data_from_pickle(num_classes):
    path = f'saved_objs/Dataset/{num_classes}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_data(dataset_num_classes=[10,15,18,20], transform=None, batch_size=128):
    datasets = []
    for num_classes in dataset_num_classes:
        train_dir = f'train_{num_classes}'
        test_dir = f'test_{num_classes}'
        dataset = Dataset(train_dir, test_dir, transform, batch_size)
        dataset.save()
        datasets.append(dataset)
    return datasets

device = get_default_device(True)

"""
subset = ['oyx', 'zmf', 'yzq', 'lgq']
# number of samples per calligrapher
n_c50 = 50
n_c100 = 100
n_c200 = 200
c = len(subset)
X_subset, y_subset = take_subset(subset, n_c)
Xsub100, ysub100 = take_subset(subset, n_c100)
Xsub200, ysub200 = take_subset(subset, n_c200)
X_red, y_red = take_subset(classes, 10)
"""