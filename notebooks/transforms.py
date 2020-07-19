from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.spatial.distance import cdist
import torch
import torchvision.transforms.functional as F
import numpy as np

class PowerTransform(object):
    
    def __init__(self, types, lambdas=[None, None], mins=[0, 0]):
        self.x_type, self.y_type = types
        self.x_lambda, self.y_lambda = lambdas
        self.x_min, self.y_min = mins
    
    def __call__(self, sample):
        x_data, y_data = sample[self.x_type], sample[self.y_type]
        
        if self.x_lambda is not None:
            x_data = boxcox(x_data-self.x_min+1, self.x_lambda)
        if self.y_lambda is not None:
            y_data = boxcox(y_data-self.y_min+1, self.y_lambda)
            
        return {self.x_type : x_data, self.y_type : y_data}
    
class InversePowerTransform(object):
    
    def __init__(self, lambda_val=None, min_val=0):
        self.lambda_val = lambda_val
        self.min_val = min_val
        
    def __call__(self, sample):
        data = sample
        if self.lambda_val is not None:
            data =  inv_boxcox(sample, self.lambda_val) + self.min_val - 1
        return data
            
class ToTensor(object):
    
    def __init__(self, types):
        self.x_type, self.y_type = types
        
    def __call__(self, sample):
        x_data, y_data = sample[self.x_type], sample[self.y_type]
        return {self.x_type : torch.tensor(x_data.copy(), dtype=torch.float32), 
                self.y_type : torch.tensor(y_data.copy(), dtype=torch.float32)}

class RandomRotatation(object):

    def __init__(self, types):
        self.x_type, self.y_type = types

    def __call__(self, sample):
        x_data, y_data = sample[self.x_type], sample[self.y_type]
        
        num_90_turns = [0, 1, 2, 3]
        rot = np.random.choice(num_90_turns, size=1)
        
        x_data = np.rot90(x_data, k=rot, axes=(1, 2))
        y_data = np.rot90(y_data, k=rot, axes=(1, 2))
        
        return {self.x_type : x_data, self.y_type : y_data}
    
class RandomFlip(object):
    
    def __init__(self, types):
        self.x_type, self.y_type = types
        
    def __call__(self, sample):
        x_data, y_data = sample[self.x_type], sample[self.y_type]
        
        do_flip = np.random.choice([True, False], size=1)
        if do_flip:
            x_data = np.flip(x_data, axis=1)
            y_data = np.flip(y_data, axis=1)
            
        return {self.x_type : x_data, self.y_type : y_data}
    
class RandomCrop(object):
    
    #Assumes that x_data and y_data have the same size
    def __init__(self, types, crop_size=(512, 512)):
        self.x_type, self.y_type = types
        self.crop_size = crop_size
        
    def __call__(self, sample):
        x_data, y_data = sample[self.x_type], sample[self.y_type]
        
        corner_x = np.randint(0, x_data.shape[0]-self.crop_size[0]+1, size=1)
        corner_y = np.randint(0, x_data.shape[1]-self.crop_size[1]+1, size=1)
        other_corner_x, other_corner_y = corner_x + crop_size[0], corner_y + crop_size[1]
        
        return {self.x_type : x_data[:, corner_x:other_corner_x, corner_y:other_corner_y],
                self.y_type : y_data[:, corner_x:other_corner_x, corner_y:other_corner_y]}
        
    
class Normalize(object):
    
    def __init__(self, types, means=[0, 0], stds=[1, 1], inplace=False):
        self.x_type, self.y_type = types
        self.x_mean, self.y_mean = means
        self.x_std, self.y_std = stds
        self.inplace = inplace
        
    def __call__(self, sample):
        x_tensor, y_tensor = sample[self.x_type], sample[self.y_type]
        
        x_tensor = F.normalize(x_tensor, self.x_mean, self.x_std, self.inplace)
        y_tensor = F.normalize(y_tensor, self.y_mean, self.y_std, self.inplace)
            
        return {self.x_type : x_tensor, self.y_type : y_tensor}
    
class InverseNormalize(object):
    
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        return sample * self.std + self.mean
    
class AddDistChannel(object):
    
    def __init__(self, types, size=(512, 512), metric='euclidean'):
        self.x_type, self.y_type = types
        self.img_size = size
        new_arr = []
        for i in range(self.img_size[0]):
            for j in range(self.img_size[1]):
                new_arr.append([i, j])
        self.dist_img = cdist(new_arr, [[(self.img_size[0]-1) / 2, (self.img_size[1]-1) / 2]], metric=metric)
        self.dist_img = self.dist_img.reshape(self.img_size) / np.max(self.dist_img)
        self.dist_img = self.dist_img[None, :, :]
        
    def get_mean_std(self):
        mean = np.mean(self.dist_img)
        std = np.std(self.dist_img)
        return mean, std
        
    def __call__(self, sample):
        x_data, y_data = sample[self.x_type], sample[self.y_type]
        x_data = np.concatenate((x_data, self.dist_img), axis=0)
        
        return {self.x_type : x_data, self.y_type : y_data}
    
    
class AddLatitudeChannel(object):
    
    def __init__(self, types, size=(512, 512)):
        self.x_type, self.y_type = types
        self.img_size = size
        self.lat_img = np.zeros(self.img_size)
        for i in range(self.img_size[0]):
            self.lat_img[i] = np.ones(self.img_size[1]) * (self.img_size[0] - i - 1)
        self.lat_img = self.lat_img - self.img_size[0] / 2
        self.lat_img = self.lat_img / np.max(self.lat_img)
        self.lat_img = self.lat_img[None, :, :]
        
    def get_mean_std(self):
        mean = np.mean(self.lat_img)
        std = np.std(self.lat_img)
        return mean, std
        
    def __call__(self, sample):
        x_data, y_data = sample[self.x_type], sample[self.y_type]
        x_data = np.concatenate((x_data, self.lat_img), axis=0)
        
        return {self.x_type : x_data, self.y_type : y_data}