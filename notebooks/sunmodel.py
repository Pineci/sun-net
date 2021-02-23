import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.stats import boxcox
import pandas as pd
from tqdm import tqdm
import h5py
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms as tvtf
from torch.utils.data import DataLoader, Subset

from unet import UNet
from fcn import FCN
from pixelwise import Pixelwise

from dataset import SunImageDataset
from early_stopping import EarlyStopping
import transforms

class SunModel(object):
    '''This class creates a model to predict EUV emissions of the sun using HEI
    absorption images.
    
    Attributes
    ----------
    
    in_feature : str
        String identifier for the input image type, this is 'ew', which is an 
        HEI absorption image
    out_feature : str
        String identifier for the output image type, this is '0304'
    device_type : str
        Type of device to do computations, can be either 'cpu' or 'gpu'
    model_type : str
        Identifier for model type, can only be 'unet'
    optimizer_type : str
        Identifier for optimizer type, can be 'RMSprop' or 'AdamW'
    scheduler_type : str
        Identifier for learning rate scheduler type, can be only 'ReduceLROnPlateau'
    save_path : pathlib.Path
        Path for saving model checkpoints
    
    val_proportion : float
        Proportion of dataset that is reserved for validation
    n_val : int
        Number of images in the validation dataset
    n_train : int
        Number of images in the training dataset
    random_state : int, None
        Random seed for splitting the dataset into training and validation sets,
        reuse the same seed for reproducability
    dataset : SunImageDataset
        A custom dataset class which manages sun images, implements the torch Dataset
        class
    train : Subset
        A subset of self.dataset which is used for training
    val : Subset
        A subset of self.dataset which is reserved for validation
    power_transforms : tuple(float)
        A tuple indicating whether a power transform will be applied to the x or y data.
        Here, None indicates no transform to be done, while a float will be used as the
        lambda parameter for a boxcox transformation
    add_distance_channel : boolean
        A boolean which indicates whether to add a second channel to the input data which
        stores the distance from the center of the image
    
    training_history : pd.DataFrame
        A dataframe which stores the loss history during training every epoch
    criterion : nn.Loss
        Loss function for the model. This is fixed to MSELoss
    last_epoch : int
        The iteration of the last epoch that was trained on
    device : torch.device
        The handle for the device specified by self.device_type
    model : nn.Module
        The handle for the model specified by self.model_type
    optimizer : optim.Optimizer
        The handle for the optimizer specified by self.optimizer_type
    scheduler : optim.lr_scheduler
        The handle for the scheduler specified by self.scheduler_type
        
    Methods
    -------
    settings_model(model, add_distance_channel, dropout)
        Specifies the model to be used and trained
    settings_criterion(criterion)
        Specifies the loss function to be used during training
    settings_optimizer(optimizer, lr, weight_decay, momentum)
        Specifies the optimizer to be used during training
    settings_scheduler(scheduler, factor, patience, min_lr)
        Specifies the learning rate scheduler for the optimizer
    create_val_train_sets(val_proportion, random_state)
        Creates the training and validation sets for the dataset
    save_model()
        Saves the current state of the model, optimizer, scheduler, and class parameters
    load_model()
        Loads the model, optimizer, scheduler, and class parameters. After loading,
        the model can resume training (if it was interrupted)
    eval_network(data_loader)
        Computes the evaluation loss of the network on the subset of data specified by
        data loader
    show_image_predictions(mode, color_limit)
        Shows images side-by-side of input, output, and predicted output images of the model
    get_image_prediction(num, mode):
        Draws a sample of images from a specified dataset and returns the relevant images
    show_training_curve():
        Plots a 
    train_network(epochs, batch_size, num_evals_per_epoch)
        Train the network up to the specified number of epochs with the given batch_size
    '''
    
    in_feature = 'in'
    out_feature = 'out'
    
    #power_transforms = (None, 0.0)
    
    def __init__(self, device='cpu', save_folder='models', save_name='model.pt', load_model=False, in_types=['ew'], out_types=['0304']):
        '''
        Parameters
        ----------
        
        device : str, optional
            Identifier for device type of the model, i.e. where the computations will be performed.
            Can be either 'cpu' or 'gpu' (default = 'cpu')
        save_folder : str, optional
            Name of the folder containing the saved models, helps to keep model files 
            organized (default = 'models')
        save_name : str, optional
            Filename of the model to be saved in the save_folder, should end with a '.pt' file 
            extension (default = 'model.pt')
        load_model : boolean, optional
            If true, loads the model specified at save_name file inside save_folder. Also loads the
            optimizer, scheduler, train/validation sets (default = False)
        '''
        print("Initializing...")
        self.save_folder = save_folder
        self.save_path = pathlib.Path(save_folder)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_path / save_name
        
        self.in_types = in_types
        self.out_types = out_types
        
        self.last_epoch = 0
        self.training_history = pd.DataFrame()
        
        #Set up device
        self.device_type = device
        if device == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        elif device == 'gpu':
            self.device = torch.device('cuda:0')
        else:
            raise ValueError('SunModel: Received device type ' + str(device) + ', must be one of: ' + str(['cpu', 'gpu']))
            
        
        
        self.loaded_model = False
        #Load model if specified
        if load_model and self.save_path.exists():
            self.load_model()
            self.loaded_model = True
            
        #Set up input data frame
        #print("Loading Dataset...")
        #self.dataset = SunImageDataset(data_source='ssd')
        
    def settings_dataset(self, in_types=['ew'], out_types=['0304']):
        self.in_types = in_types
        self.out_types = out_types
        self.dataset = SunImageDataset(in_types=in_types, out_types=out_types, data_source='ssd')

    def settings_model(self, model='unet', add_distance_channel=False, add_latitude_channel=False, dropout=None, use_batch_norm=False):
        '''This function specifies the model to be used. Currently, the only available model is 'unet',
        or 'unet-dropout'.
        
        Parameters
        ----------
        
        model : str, optional
            Identifier for the model type to be used (default = 'unet')
        add_distance_channel : boolean, optional
            Boolean which indicates whether to add a distance channel to the input images. This channel
            could help the network to isolate edge effects on the target.
        dropout : float, None, optional
            If not None, specifies the dropout probability. Only used for 'unet-dropout' model type.
            (default = None)
        
        '''
        self.model_type = model
        self.add_distance_channel = add_distance_channel
        self.add_latitude_channel = add_latitude_channel
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        in_channels = len(self.in_types)
        if self.add_distance_channel:
            in_channels += 1
        if self.add_latitude_channel:
            in_channels += 1
        out_channels = len(self.out_types)
            
        if model == 'unet':
            self.model = UNet(in_channels=in_channels, out_channels=out_channels, 
                              bilinear=False, dropout=dropout)
        elif model == 'fcn8':
            self.model = FCN('8', in_channels=in_channels, out_channels=out_channels, 
                             single_pixel_pred=False, use_batch_norm=use_batch_norm, dropout=dropout)
        elif model == 'fcn16':
            self.model = FCN('16', in_channels=in_channels, out_channels=out_channels, 
                             single_pixel_pred=False, use_batch_norm=use_batch_norm, dropout=dropout)
        elif model == 'fcn32':
            self.model = FCN('32', in_channels=in_channels, out_channels=out_channels, 
                             single_pixel_pred=False, use_batch_norm=use_batch_norm, dropout=dropout)
        elif model == 'fcn8-single':
            self.model = FCN('8', in_channels=in_channels, out_channels=out_channels, 
                             single_pixel_pred=True, use_batch_norm=use_batch_norm, dropout=dropout)
        elif model == 'fcn16-single':
            self.model = FCN('16', in_channels=in_channels, out_channels=out_channels, 
                             single_pixel_pred=True, use_batch_norm=use_batch_norm, dropout=dropout)
        elif model == 'fcn32-single':
            self.model = FCN('32', in_channels=in_channels, out_channels=out_channels, 
                             single_pixel_pred=True, use_batch_norm=use_batch_norm, dropout=dropout)
        elif model == 'pixelwise':
            self.model = Pixelwise(in_channels=in_channels, out_channels=out_channels)
        else:
            raise ValueError('SunModel: Received model type ' + str(model) + ', must be one of: ' + str(['unet', 'fcn8', 'fcn16', 'fcn32']))
        self.model = self.model.to(self.device)
        self.best_model_params = self.model.state_dict()
        
    def settings_criterion(self, criterion='MSE'):
        '''This function specifies the loss function to be used during training.
        
        Parameters
        ----------
        
        criterion : str, optional
            The name of the loss function to be used. Options include 'MSE' and 'MAE' (default = 'MSE')
        '''
        self.criterion_type = criterion
        if criterion == 'MSE':
            self.criterion = nn.MSELoss(reduction='sum')
        elif criterion == 'MAE':
            self.criterion = nn.L1Loss(reduction='sum')
        else:
            raise ValueError('SunModel: Received criterion type ' + str(criterion) + ', must be one of: ' + str(['MSE, MAE']))
        
        
    def settings_optimizer(self, optimizer='AdamW', lr=0.001, weight_decay=1e-7, momentum=0.8, batch_size=4, num_steps_per_batch=1):
        '''This function specifies the optimizer to be used during training. Currently, the 
        available optimizers are RMSprop and AdamW.
        
        Parameters
        ----------
        
        optimizer : str, optional
            Identifier for optimizer type to be used (default = 'AdamW')
        lr : float, optinal
            Initial learning rate for the optimizer (default = 0.001)
        weight_decay : float, optional
            Weight decay parameter for optimizer (default = 1e-7)
        momentum : float, optonal
            Momentum parameter. Only used for RMSProp (default = 0.8)
        
        '''
        self.optimizer_type = optimizer
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('SunModel: Received optimizer type ' + str(optimizer) + ', must be one of: ' + str(['RMSprop, AdamW']))
        self.optimizer_lr = lr
        self.optimizer_weight_decay = weight_decay
        self.optimizer_momentum = momentum
        self.batch_size = batch_size
        self.num_steps_per_batch = num_steps_per_batch
        
        
    def settings_scheduler(self, scheduler='ReduceLROnPlateau', factor=0.1, patience=2, min_lr=1e-8):
        '''This function specifies the scheduler to be used during training. Currently, the 
        only available scheduler is ReduceLROnPlateau.
        
        Parameters
        ----------
        
        scheduler : str, optional
            Identifier for optimizer type to be used (default = 'ReduceLROnPlateau')
        factor : float, optional
            Factor by which to reduce the learning rate according to the scheduler (default = 0.1)
        patience : int, optional
            Number of epochs to wait for seeing no improvement in validation score to decrease
            the learning rate by factor. Used by ReduceLROnPlateau (default = 2)
        min_lr : float, optional
            A floor for the learning rate, so that it cannot be decreased beneath this value (default = 1e-8)
        
        '''
        self.scheduler_type = scheduler
        if scheduler == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=factor, patience=patience, min_lr=min_lr)
        else:
            raise ValueError('SunModel: Received scheduler type ' + str(scheduler) + ', must be one of: ' + str(['ReduceLROnPlateau']))
        self.scheduler_factor = factor
        self.scheduler_patience = patience
        self.scheduler_min_lr = min_lr
        
    def settings_early_stopping(self, patience=10, mode='min', min_delta=0, percentage=False):
        self.early_stopping_patience = patience
        self.early_stopping_mode = mode
        self.early_stopping_min_delta = min_delta
        self.early_stopping_percentage = percentage
        self.early_stopping = EarlyStopping(patience=patience, mode=mode, min_delta=min_delta, percentage=percentage)
    
        
    def create_train_val_test_sets(self, val_proportion=0.2, test_proportion=0.2, random_state=None, power_transforms=[[None], [0.0]], train_stats_dict=None, train_idx=None, val_idx=None, test_idx=None, mode='random'):
        '''Creates training and validation sets for the model to train on. Uses the
        random_state as a seed to shuffle the indices, and can be set to the same value
        for reproducability. The val_proportion variable can be used to specify the 
        proportion of the data used for validation. The inputs and outputs are normalized
        according to the mean and std of the training set, and this transform is applied
        to both the training and validation sets.
        
        Parameters
        ----------
        
        val_proportion : float, optional
            Proportion of the data used for validation (default = 0.2)
        random_state : int, None, optional
            Seed for the random number generator. If None, a random seed is selected (default = None)
        '''
        print('Setting up training set...')
        
        
        def get_all_dates():
            times = self.dataset.get_date(list(range(len(self.dataset))))
            return list(map(lambda x: datetime.fromtimestamp(x), times))
        
        #train_idx, val_idx, test_idx = None, None, None
        if mode != 'random':
            dates=get_all_dates()
            train_idx, val_idx, test_idx = [], [], []
            if mode == 'month-regular': #Jan-June is train, July-Sep valid, Oct-Dec test
                for i in range(len(dates)):
                    month = dates[i].month
                    if month <= 6:
                        train_idx.append(i)
                    elif month >= 7 and month <= 9:
                        val_idx.append(i)
                    else:
                        test_idx.append(i)
            if mode == 'month':
                years = list(map(lambda x: x.year, dates))
                min_yr, max_yr = min(years), max(years)
                month_designations = {}
                np.random.seed(random_state)
                for yr in range(min_yr, max_yr+1):
                    months = ['train'] * 6 + ['valid']*3 + ['test']*3
                    np.random.shuffle(months)
                    month_designations[yr] = months
                #print(months.keys())
                for i in range(len(dates)):
                    #print(dates[i].month-1)
                    designation = month_designations[dates[i].year][dates[i].month-1]
                    if designation == 'train':
                        train_idx.append(i)
                    elif designation == 'valid':
                        val_idx.append(i)
                    elif designation == 'test':
                        test_idx.append(i)
                
                
        
        self.best_score = np.inf
        self.val_proportion = val_proportion
        self.test_proportion = test_proportion
        self.random_state = random_state
        if not isinstance(power_transforms[0], list) or not isinstance(power_transforms[1], list):
            power_transforms = [[power_transforms[0]], [power_transforms[1]]]
        self.power_transforms = power_transforms
        n_tot = len(self.dataset)
        n_val = int(n_tot * self.val_proportion)
        n_test = int(n_tot * self.test_proportion)
        n_train = n_tot - (n_val + n_test)
        self.n_val = n_val
        self.n_train = n_train
        self.n_test = n_test
        indices = list(range(n_tot))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        train_idx_t, val_idx_t, test_idx_t = indices[:n_train], indices[n_train:n_val+n_train], indices[n_val+n_train:n_tot]
        
        print("Random State: {}".format(random_state))
        
        self.train_idx = np.sort(train_idx_t)
        self.val_idx = np.sort(val_idx_t)
        self.test_idx = np.sort(test_idx_t)
        
        if train_idx is not None and val_idx is not None and test_idx is not None:
            self.train_idx = np.sort(train_idx)
            self.val_idx = np.sort(val_idx)
            self.test_idx = np.sort(test_idx)
        
        if train_stats_dict is None or True:
            #train_tensor_dict = self.dataset[self.train_idx]
            #x_train_data = train_tensor_dict[self.in_feature]
            #y_train_data = train_tensor_dict[self.out_feature]

            metric_eval_batch_size = 100
            #x_train_min, y_train_min = np.Inf, np.Inf
            x_train_min, y_train_min = None, None
            x_train_max, y_train_max = None, None
            x_positive, x_negative, y_positive, y_negative = 0, 0, 0, 0
            
            threshold_x = -4662.5
            threshold_y = -82.43375
            
            for i in range(0, n_tot, metric_eval_batch_size):
                data_slice = self.dataset[i:(i+metric_eval_batch_size)]
                slice_x_mins = np.min(data_slice[self.in_feature], axis=(0, 2, 3))
                slice_y_mins = np.min(data_slice[self.out_feature], axis=(0, 2, 3))
                slice_x_maxs = np.max(data_slice[self.in_feature], axis=(0, 2, 3))
                slice_y_maxs = np.max(data_slice[self.out_feature], axis=(0, 2, 3))
                
                
                x_positive += np.sum(data_slice[self.in_feature] >= threshold_x)
                x_negative += np.sum(data_slice[self.in_feature] < threshold_x)
                y_positive += np.sum(data_slice[self.out_feature] >= threshold_y)
                y_negative += np.sum(data_slice[self.out_feature] < threshold_y)
                
                if x_train_min is None:
                    x_train_min = slice_x_mins
                else:
                    x_train_min = np.min(np.concatenate((x_train_min[:, None], slice_x_mins[:, None]), axis=1), axis=1)
                if x_train_max is None:
                    x_train_max = slice_x_maxs
                else:
                    x_train_max = np.min(np.concatenate((x_train_max[:, None], slice_x_maxs[:, None]), axis=1), axis=1)
                    
                if y_train_min is None:
                    y_train_min = slice_y_mins
                else:
                    y_train_min = np.min(np.concatenate((y_train_min[:, None], slice_y_mins[:, None]), axis=1), axis=1)
                if y_train_max is None:
                    y_train_max = slice_y_maxs
                else:
                    y_train_max = np.min(np.concatenate((y_train_max[:, None], slice_y_maxs[:, None]), axis=1), axis=1)
                #x_train_min = np.min((x_train_min, np.min(data_slice[self.in_feature])))
                #y_train_min = np.min((y_train_min, np.min(data_slice[self.out_feature])))
                
            #print('X Minimum: {}'.format(x_train_min))
            #print('Y Minimum: {}'.format(y_train_min))
            #print('X Maximum: {}'.format(x_train_max))
            #print('Y Maximum: {}'.format(y_train_max))
            
            #print('X Positive Count: {}'.format(x_positive))
            #print('X Negative Count: {}'.format(x_negative))
            #print('Y Positive Count: {}'.format(y_positive))
            #print('Y Negative Count: {}'.format(y_negative))
            
            #print('X Ratio to exclude: {}'.format(x_negative/(x_negative+x_positive)))
            #print('Y Ratio to exclude: {}'.format(y_negative/(y_negative+y_positive)))
            
            print(x_train_min)
            
            threshold_x = [threshold_x]
            threshold_y = [threshold_y]

            #x_transform, y_transform = self.power_transforms
            power_trans = transforms.PowerTransform([self.in_feature, self.out_feature],
                                                    lambdas=self.power_transforms, 
                                                    mins=[threshold_x, threshold_y])
            x_vals, x2_vals, y_vals, y2_vals = None, None, None, None
            n_x_vals, n_y_vals = None, None
            for i in range(0, n_train, metric_eval_batch_size):
                train_slice = self.dataset[self.train_idx[i:(i+metric_eval_batch_size)]]
                #print(train_slice[self.in_feature].shape)
                #print(train_slice[self.out_feature].shape)
                train_slice = power_trans(train_slice)
                x_train_slice = train_slice[self.in_feature]
                y_train_slice = train_slice[self.out_feature]

                #if x_transform is not None:
                #    x_train_slice = np.log(x_train_slice - x_train_min + 1)
                    #x_train_slice = boxcox(np.flatten(x_train_slice) - x_train_min + 1, x_transform)
                #if y_transform is not None:
                #    y_train_slice = np.log(y_train_slice - y_train_min + 1)
                    #y_train_slice = boxcox(np.flatten(y_train_slice) - y_train_min + 1, y_transform)
                

                shape_x, shape_y = x_train_slice.shape, y_train_slice.shape
                if n_x_vals is None:
                    n_x_vals = np.zeros(shape=(shape_x[1]))
                if n_y_vals is None:
                    n_y_vals = np.zeros(shape=(shape_y[1]))
                n_x_vals = n_x_vals + shape_x[0] * shape_x[2] * shape_x[3]
                n_y_vals = n_y_vals + shape_y[0] * shape_y[2] * shape_y[3]

                if x_vals is None:
                    x_vals = np.sum(x_train_slice, axis=(0, 2, 3))
                else:
                    x_vals = x_vals + np.sum(x_train_slice, axis=(0, 2, 3))
                if y_vals is None:
                    y_vals = np.sum(y_train_slice, axis=(0, 2, 3))
                else:
                    y_vals = y_vals + np.sum(y_train_slice, axis=(0, 2, 3))

                if x2_vals is None:
                    x2_vals = np.sum(np.square(x_train_slice), axis=(0, 2, 3))
                else:
                    x2_vals = x2_vals + np.sum(np.square(x_train_slice), axis=(0, 2, 3))
                if y2_vals is None:
                    y2_vals = np.sum(np.square(y_train_slice), axis=(0, 2, 3))
                else:
                    y2_vals = y2_vals + np.sum(np.square(y_train_slice), axis=(0, 2, 3))
                #print(x_vals)
                #print(y_vals)

            x_train_mean = x_vals / n_x_vals
            y_train_mean = y_vals / n_y_vals
            x_train_std = np.sqrt(x2_vals / n_x_vals + np.square(x_train_mean))
            y_train_std = np.sqrt(y2_vals / n_y_vals + np.square(y_train_mean))
            
            self.train_stats_dict = {}
            self.train_stats_dict['x_train_min'] = threshold_x
            self.train_stats_dict['y_train_min'] = threshold_y
            #self.train_stats_dict['x_train_min'] = x_train_min
            #self.train_stats_dict['y_train_min'] = y_train_min
            self.train_stats_dict['x_train_mean'] = x_train_mean
            self.train_stats_dict['y_train_mean'] = y_train_mean
            self.train_stats_dict['x_train_std'] = x_train_std
            self.train_stats_dict['y_train_std'] = y_train_std
            print(self.train_stats_dict)
        else:
            
            if not isinstance(train_stats_dict['x_train_min'], np.ndarray):
                train_stats_dict['x_train_min'] = np.array(train_stats_dict['x_train_min'])[None]
                train_stats_dict['y_train_min'] = np.array(train_stats_dict['y_train_min'])[None]
                train_stats_dict['x_train_mean'] = np.array(train_stats_dict['x_train_mean'])[None]
                train_stats_dict['y_train_mean'] = np.array(train_stats_dict['y_train_mean'])[None]
                train_stats_dict['x_train_std'] = np.array(train_stats_dict['x_train_std'])[None]
                train_stats_dict['y_train_std'] = np.array(train_stats_dict['y_train_std'])[None]
            
            
            x_train_min = train_stats_dict['x_train_min']
            y_train_min = train_stats_dict['y_train_min']
            x_train_mean = train_stats_dict['x_train_mean']
            y_train_mean = train_stats_dict['y_train_mean']
            x_train_std = train_stats_dict['x_train_std']
            y_train_std = train_stats_dict['y_train_std']
            self.train_stats_dict = train_stats_dict
            print(self.train_stats_dict)
            
        #print(self.train_stats_dict)
        
        features = [self.in_feature, self.out_feature]
        self.transform_train = tvtf.Compose(self.generate_transform(features, mins=[x_train_min, y_train_min],
                                                                    means=[x_train_mean, y_train_mean],
                                                                    stds=[x_train_std, y_train_std],
                                                                    add_rotation=True, 
                                                                    add_distance=self.add_distance_channel,
                                                                    add_latitude=self.add_latitude_channel))
        self.transform_eval = tvtf.Compose(self.generate_transform(features, mins=[x_train_min, y_train_min],
                                                                   means=[x_train_mean, y_train_mean],
                                                                   stds=[x_train_std, y_train_std],
                                                                   add_rotation=False, 
                                                                   add_distance=self.add_distance_channel,
                                                                   add_latitude=self.add_latitude_channel))
        self.transform_output_inverse = self.generate_transform(features, mins=[x_train_min, y_train_min],
                                                                means=[x_train_mean, y_train_mean],
                                                                stds=[x_train_std, y_train_std],
                                                                inverse=True)
        
        self.dataset.set_transform(transform=self.transform_train)
        train, val, test = Subset(self.dataset, indices=self.train_idx), Subset(self.dataset, indices=self.val_idx), Subset(self.dataset, indices=self.test_idx)
        self.train = train
        self.val = val
        self.test = test
        
    def generate_transform(self, features, mins=[[0], [0]], means=[[0], [0]], stds=[[1], [1]], add_rotation=True, add_distance=False, add_latitude=False, inverse=False):
        transform_list = []
        
        if not inverse:
            transform_list.append(transforms.PowerTransform(features, 
                                                        lambdas=self.power_transforms, 
                                                        mins=mins))
            
            #if add_distance or add_latitude:
            #    means[0] = [means[0]]
            #    stds[0] = [stds[0]]
            

            if add_distance:
                transform_dist_channel = transforms.AddDistChannel(features)
                transform_list.append(transform_dist_channel)
                dist_mean, dist_std = transform_dist_channel.get_mean_std()
                means[0] = np.concatenate((means[0], np.array(dist_mean)[None]))
                stds[0] = np.concatenate((stds[0], np.array(dist_std)[None]))
            
            if add_latitude:
                transform_lat_channel = transforms.AddLatitudeChannel(features)
                transform_list.append(transform_lat_channel)
                lat_mean, lat_std = transform_lat_channel.get_mean_std()
                means[0] = np.concatenate((means[0], np.array(lat_mean)[None]))
                stds[0] = np.concatenate((stds[0], np.array(lat_std)[None]))
                
            if add_rotation:
                transform_list.append(transforms.RandomRotatation(features))
                transform_list.append(transforms.RandomFlip(features))
                transform_list.append(transforms.RandomCrop(features))
            
            transform_list.append(transforms.ToTensor(features))
            transform_list.append(transforms.Normalize(features, means=means, stds=stds))   
        else:
            transform_list.append(transforms.InverseNormalize(mean=means[1], std=stds[1]))
            transform_list.append(transforms.InversePowerTransform(lambda_val=self.power_transforms[1], min_val=mins[1]))
            
        return transform_list
        
    def save_model(self):
        '''Saves the model state and the state of the optimizer and scheduler. Also saves
        the number of epochs that have been trained on as well as the model properties.
        This saves all necessary information to reload the model and resume training
        from only the saved file specified by self.save_path (made in the constructor).
        
        '''
        
        model_properties = {}
        
        model_properties['in_types'] = self.in_types
        model_properties['out_types'] = self.out_types
        
        model_properties['model_type'] = self.model_type
        model_properties['add_distance_channel'] = self.add_distance_channel
        model_properties['add_latitude_channel'] = self.add_latitude_channel
        model_properties['dropout'] = self.dropout
        model_properties['use_batch_norm'] = self.use_batch_norm
        
        model_properties['optimizer_type'] = self.optimizer_type
        model_properties['optimizer_lr'] = self.optimizer_lr
        model_properties['optimizer_weight_decay'] = self.optimizer_weight_decay
        model_properties['optimizer_momentum'] = self.optimizer_momentum
        model_properties['batch_size'] = self.batch_size
        model_properties['num_steps_per_batch'] = self.num_steps_per_batch
        
        model_properties['scheduler_type'] = self.scheduler_type
        model_properties['scheduler_factor'] = self.scheduler_factor
        model_properties['scheduler_patience'] = self.scheduler_patience
        model_properties['scheduler_min_lr'] = self.scheduler_min_lr
        
        model_properties['criterion_type'] = self.criterion_type
        
        model_properties['early_stopping_mode'] = self.early_stopping_mode
        model_properties['early_stopping_patience'] = self.early_stopping_patience
        model_properties['early_stopping_percentage'] = self.early_stopping_percentage
        model_properties['early_stopping_min_delta'] = self.early_stopping_min_delta
        model_properties['early_stopping_state'] = self.early_stopping.get_state()
        
        model_properties['val_proportion'] = self.val_proportion
        model_properties['test_proportion'] = self.test_proportion
        model_properties['random_state'] = self.random_state        
        model_properties['power_transforms'] = self.power_transforms
        model_properties['train_stats_dict'] = self.train_stats_dict
        model_properties['train_idx'] = self.train_idx
        model_properties['val_idx'] = self.val_idx
        model_properties['test_idx'] = self.test_idx
        
        checkpoint = {}
        checkpoint['epoch'] = self.last_epoch
        checkpoint['training_history'] = self.training_history
        checkpoint['model'] = self.model.state_dict()
        checkpoint['best_model'] = self.best_model_params
        checkpoint['best_score'] = self.best_score
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['scheduler'] = self.scheduler.state_dict()
        checkpoint['model_properties'] = model_properties
        torch.save(checkpoint, self.save_path)
        
    def load_model(self):
        '''Loads the model from the specified file from self.save_path (made in the constructor)
        All states are loaded such that training may resume from where it was left off.
        
        
        '''
        checkpoint = torch.load(self.save_path, map_location=self.device)

        model_properties = checkpoint['model_properties']
        
        print("Loading Dataset...")
        if 'in_types' in model_properties.keys() and 'out_types' in model_properties.keys():
            #print("HERE")
            self.settings_dataset(in_types=model_properties['in_types'],
                                  out_types=model_properties['out_types'])
        else:
            self.settings_dataset()
            
        print("Loading Model Settings...")
        self.settings_model(model=model_properties['model_type'], 
                            add_distance_channel=model_properties['add_distance_channel'],
                            add_latitude_channel=model_properties['add_latitude_channel'],
                            dropout=model_properties['dropout'],
                            use_batch_norm=model_properties['use_batch_norm'])
        
        print("Loading Optimizer Settings...")
        self.settings_optimizer(optimizer=model_properties['optimizer_type'],
                                lr=model_properties['optimizer_lr'],
                                weight_decay=model_properties['optimizer_weight_decay'],
                                momentum=model_properties['optimizer_momentum'],
                                batch_size=model_properties['batch_size'],
                                num_steps_per_batch=model_properties['num_steps_per_batch'])
        
        print("Loading Scheduler Settings...")
        self.settings_scheduler(scheduler=model_properties['scheduler_type'],
                                factor=model_properties['scheduler_factor'],
                                patience=model_properties['scheduler_patience'],
                                min_lr=model_properties['scheduler_min_lr'])
        
        print("Loading Criterion Settings...")
        self.settings_criterion(criterion=model_properties['criterion_type'])
        
        print("Loading Early Stopping Settings...")
        self.settings_early_stopping(mode=model_properties['early_stopping_mode'],
                                     min_delta=model_properties['early_stopping_min_delta'], 
                                     patience=model_properties['early_stopping_patience'],
                                     percentage=model_properties['early_stopping_percentage'])
        self.early_stopping.load_state(model_properties['early_stopping_state'])
        
        print("Loading Training Set Settings...")
        if 'train_idx' in model_properties:
            train_idx = model_properties['train_idx']
            val_idx = model_properties['val_idx']
            test_idx = model_properties['test_idx']
        else:
            train_idx, val_idx, test_idx = None, None, None
        self.create_train_val_test_sets(val_proportion=model_properties['val_proportion'], 
                                        test_proportion=model_properties['test_proportion'], 
                                        random_state=model_properties['random_state'],
                                        power_transforms=model_properties['power_transforms'],
                                        #train_stats_dict=None)
                                        train_stats_dict=model_properties['train_stats_dict'],
                                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        
       
        print("Loading Model...")
        self.training_history = checkpoint['training_history']
        self.last_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_model_params = checkpoint['best_model']
        self.best_score = checkpoint['best_score']
        
    def get_mask_loc(self):
        if not hasattr(self, 'mask_idx'):
            if 'ew' in self.in_types:
                self.mask_list = self.in_feature
                list_to_check = self.in_types
            else:
                self.mask_list = self.out_feature
                list_to_check = self.out_types
            for i in range(len(list_to_check)):
                if list_to_check[i] == 'ew':
                    self.mask_idx = i
        return self.mask_list, self.mask_idx
        
        
    def eval_network(self, data_loader):
        '''Evaluates the model on the instances specified in data_loader. Evaluates these
        instances in eval mode. The resulting loss is averaged across all instances
        
        Parameters
        ----------
        
        data_loader : DataLoader
            A DataLoader type which contains the data to be evaluated
        
        Returns
        -------
        
        loss : float
            The loss averaged across all instances in data_loader, uses self.criterion
            to evaluate the loss
        
        '''
        self.model.eval()
        
        total_loss = 0
        n = len(data_loader.dataset)
        for batch in data_loader:
            in_imgs = batch[self.in_feature]
            out_imgs = batch[self.out_feature]
            in_imgs = in_imgs.to(self.device)
            out_imgs = out_imgs.to(self.device)
            
            mask_data_type, mask_idx = self.get_mask_loc()
            if mask_data_type == self.in_feature:
                mask = in_imgs[:, mask_idx:(mask_idx+1), :, :] != in_imgs[0, mask_idx, 0, 0]
            else:
                mask = out_imgs[:, mask_idx:(mask_idx+1), :, :] != in_imgs[0, mask_idx, 0, 0]
            
            with torch.no_grad():
                out_pred = self.model(in_imgs)
            loss = self.criterion(out_pred * mask, out_imgs * mask) / torch.sum(mask)
            total_loss += loss.item()
        self.model.train()
        return total_loss / n
    
    def eval_network_ME(self, mode='test', metrics=['ME', 'RE']):
        if mode == 'train':
            data_loader = DataLoader(self.train, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'val':
            data_loader = DataLoader(self.val, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'test':
            data_loader = DataLoader(self.test, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'all':
            data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, pin_memory=True)
        else:
            raise ValueError("SunModel: Received invalid image dataset type")
        
        total_loss = {}
        for metric in metrics:
            total_loss[metric] = 0
        self.model.eval()
        n = len(data_loader.dataset)
        for batch in data_loader:
            in_imgs = batch[self.in_feature]
            out_imgs = batch[self.out_feature]
            in_imgs = in_imgs.to(self.device)
            out_imgs = out_imgs.to(self.device)
            
            mask_data_type, mask_idx = self.get_mask_loc()
            if mask_data_type == self.in_feature:
                mask = in_imgs[:, mask_idx:(mask_idx+1), :, :] != in_imgs[0, mask_idx, 0, 0]
            else:
                mask = out_imgs[:, mask_idx:(mask_idx+1), :, :] != in_imgs[0, mask_idx, 0, 0]
            
            with torch.no_grad():
                out_pred = self.model(in_imgs)
                
            diff = out_pred * mask - out_imgs * mask
            for metric in metrics:
                if metric == 'ME':
                    total_loss[metric] += torch.sum(torch.abs(diff)) / torch.sum(mask)
                elif metric == 'RE':
                    re_tensor = torch.abs(diff / (out_imgs * mask))
                    re_tensor[torch.isnan(re_tensor)] = 0
                    total_loss[metric] += torch.sum(re_tensor) / torch.sum(mask)
                elif metric == 'MSE':
                    mse = nn.MSELoss(reduction='sum')
                    loss = mse(out_pred * mask, out_imgs * mask) / torch.sum(mask)
                    total_loss[metric] += loss.item()
                elif metric == 'MAE':
                    mae = nn.L1Loss(reduction='sum')
                    loss = mae(out_pred * mask, out_imgs * mask) / torch.sum(mask)
                    total_loss[metric] += loss.item()
                
                    
            #loss = self.criterion(out_pred * mask, out_imgs * mask) / torch.sum(mask)
            #total_loss += loss.item()
        self.model.train()
        for metric in metrics:
            total_loss[metric] = total_loss[metric] / n
        return total_loss
    
    def show_image_predictions(self, mode='all', color_limits=[(-2, 2), (-1.5, 1.5)]):
        '''A useful visualization for the outputs of the model.
        Plots 3 images for each instance in the desired dataset, the leftmost
        image being the input image, the middle image being the output, and rightmost
        being the model's predicted output image.
        
        Parameters
        ----------
        
        mode : str, optional
            Specifies which data to be visualized. Options include 'train' for training
            data, 'val' for validation data, or 'all' for both
        color_limit : list(tuple(float)), optional
            Specifies the color bar range to be plotted across the graphs. The first graph
            has its own color scale, while the second two have the same color scale. The first
            entry in the list is the range of the first color scale while the second entry is
            the range of the second color scale. (default = [(-2, 2), (-1.5, 1.5)])
        '''
        self.model.eval()
        self.dataset.set_transform(transform=self.transform_eval)
        if mode == 'train':
            data_loader = DataLoader(self.train, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'val':
            data_loader = DataLoader(self.val, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'test':
            data_loader = DataLoader(self.test, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'all':
            data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, pin_memory=True)
        else:
            raise ValueError("SunModel: Received invalid image dataset type")
        
        if self.add_distance_channel and False:
            columns = 4
            dist_idx, in_idx, out_idx, out_pred_idx = [1, 2, 3, 4]
            
        else:
            columns = 3
            in_idx, out_idx, out_pred_idx = [1, 2, 3]
        
        rows = 1
        
        print('Left to Right: Input, Output, Predicted')
        
        for batch in data_loader:
            in_img = np.array(batch[self.in_feature])[0, 0, :, :]
            if self.add_distance_channel:
                dist_img = np.array(batch[self.in_feature])[0, 1, :, :]
            out_img = np.array(batch[self.out_feature])[0, 0, :, :]
            in_img_t = batch[self.in_feature]
            in_img_t = in_img_t.to(self.device)
            
            with torch.no_grad():
                out_pred = self.model(in_img_t)
            
            out_pred = np.array(out_pred.cpu())[0, 0, :, :]
        
            fig=plt.figure(figsize=(8, 8))
            
            if self.add_distance_channel and False:
                ax = fig.add_subplot(rows, columns, dist_idx)
                ax.set_yticklabels([])
                #ax.set_xticklabels([])
                plt.imshow(dist_img.reshape[0])
                
            ax = fig.add_subplot(rows, columns, in_idx)
            ax.set_yticklabels([])
            #ax.set_xticklabels([])
            plt.imshow(in_img)
            plt.clim(color_limits[0][0], color_limits[0][1])
            
            ax = fig.add_subplot(rows, columns, out_idx)
            ax.set_yticklabels([])
            #ax.set_xticklabels([])
            plt.imshow(out_img)
            plt.clim(color_limits[1][0], color_limits[1][1])
            
            ax = fig.add_subplot(rows, columns, out_pred_idx)
            ax.set_yticklabels([])
            #ax.set_xticklabels([])
            plt.imshow(out_pred)
            plt.clim(color_limits[1][0], color_limits[1][1])
            plt.show()
        self.model.train()
        self.dataset.set_transform(transform=self.transform_train)
                               
    def get_image_prediction(self, hf_filename, mode='all', include_original=True):
        '''Gets the input, output, and predicted image from a specified dataset. A 
        specified number of images are drawn randomly from the specified dataset.
        
        Parameters
        ----------
        num_images : int
            Number of images to be sampled and predicted from the dataset
        mode : str, optional
            Dataset from which the images are accessed, could be 'all', 'train', 'val'
            
        Returns
        -------
        
        image_df : pd.DataFrame
            DataFrame containing the input, output, and predicted images
        
        '''
        
        self.model.eval()
        self.dataset.set_transform(transform=self.transform_eval)
        if mode == 'train':
            data_loader = DataLoader(self.train, batch_size=1, shuffle=False, pin_memory=True)
            n = self.n_train
        elif mode == 'val':
            data_loader = DataLoader(self.val, batch_size=1, shuffle=False, pin_memory=True)
            n = self.n_val
        elif mode == 'test':
            data_loader = DataLoader(self.test, batch_size=1, shuffle=False, pin_memory=True)
            n = self.n_test
        elif mode == 'all':
            data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, pin_memory=True)
            n = self.n_train + self.n_val + self.n_test
            
        #image_df = pd.DataFrame()
        
        sample = self.dataset[0]
        #in_size = (sample[self.in_feature].shape[1], sample[self.in_feature].shape[2])
        out_size = (sample[self.out_feature].shape[1], sample[self.out_feature].shape[2])
        
        #data_in_shape = (n, in_size[0], in_size[1])
        data_out_shape = (n, out_size[0], out_size[1])
        with h5py.File(hf_filename, 'w') as hf:
            hf.create_dataset('predicted', shape=data_out_shape, dtype=np.float32)
            hf.create_dataset('idx', shape=(n,))
            hf.create_dataset('type', shape=(n, 1), dtype=h5py.string_dtype())
        
        idx = 0
        with h5py.File(hf_filename, 'a') as hf:
            for batch in data_loader:
                in_img = np.array(batch[self.in_feature])
                out_img = np.array(batch[self.out_feature])
                in_img_t = batch[self.in_feature]
                in_img_t = in_img_t.to(self.device)

                mask_data_type, mask_idx = self.get_mask_loc()
                if mask_data_type == self.in_feature:
                    mask = in_img[:, mask_idx:(mask_idx+1), :, :] != in_img[0, mask_idx, 0, 0]
                else:
                    mask = out_img[:, mask_idx:(mask_idx+1), :, :] != in_img[0, mask_idx, 0, 0]

                with torch.no_grad():
                    out_pred = self.model(in_img_t)

                out_pred = np.array(out_pred.cpu())

                out_img = out_img * mask
                out_pred = out_pred * mask

                if idx in self.train_idx:
                    data_type = 'train'
                elif idx in self.val_idx:
                    data_type = 'valid'
                elif idx in self.test_idx:
                    data_type = 'test'

                '''
                if include_original:
                    new_entry = pd.Series({'idx': idx,
                                           'type': data_type,
                                           self.in_feature: in_img[0], 
                                           self.out_feature: out_img[0], 
                                           'predicted': out_pred[0]})
                else:
                    new_entry = pd.Series({'idx': idx, 
                                           'type': data_type,
                                           'predicted': out_pred[0]})
                image_df = image_df.append(new_entry, ignore_index=True)
                '''
                hf['predicted'][idx] = out_pred[0]
                hf['idx'][idx] = idx
                hf['type'][idx] = data_type
                idx += 1

            self.model.train()
            self.dataset.set_transform(transform=self.transform_train)
            #return image_df
    
    def show_training_curve(self):
        epochs = self.training_history['epoch']
        training_loss = self.training_history['train loss']
        val_loss = self.training_history['validation loss']
        plt.plot(epochs, training_loss, label="train")
        plt.plot(epochs, val_loss, label="validation")
        plt.legend(loc="best")
        #plt.ylim(-1.5, 2.0)
        plt.show()
        
            
    def train_network(self, epochs=10):
        '''Trains the model using the specified number of epochs and batch size. Uses
        the preconfigured model, optimizer, scheduler, and train/validation sets. 
        
        Parameters
        ----------
        
        epochs : int, optional
            The number of epochs to train for
        batch_size : int, optional
            The number of instances in every batch
        '''
        train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(self.val, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        
        print('BEGIN MODEL TRAINING')
        print('Model Training head: {}'.format(self.train_idx[:10]))
        print('Model Validation head: {}'.format(self.val_idx[:10]))
        print('Model Testing head: {}'.format(self.test_idx[:10]))
        
        done = self.early_stopping.step(None, check=True)
        while self.last_epoch < epochs and not done:
            print('EPOCH: ' + str(self.last_epoch), flush=True)
            self.model.train()
            epoch_loss, batch_loss, num_steps = 0, 0, 0
            with tqdm(total=len(train_loader), desc='Batches Completed') as pbar:
                for batch in train_loader:
                    in_imgs = batch[self.in_feature]
                    out_imgs = batch[self.out_feature]
                    in_imgs = in_imgs.to(self.device)
                    out_imgs = out_imgs.to(self.device)

                    mask_data_type, mask_idx = self.get_mask_loc()
                    if mask_data_type == self.in_feature:
                        mask = in_imgs[:, mask_idx:(mask_idx+1), :, :] != in_imgs[0, mask_idx, 0, 0]
                    else:
                        mask = out_imgs[:, mask_idx:(mask_idx+1), :, :] != in_imgs[0, mask_idx, 0, 0]

                    out_pred = self.model(in_imgs)
                    loss = self.criterion(out_pred * mask, out_imgs * mask) / torch.sum(mask)
                    batch_loss = batch_loss + loss
                    epoch_loss += loss.item()

                    num_steps += 1
                    if num_steps % self.num_steps_per_batch == 0 or num_steps == len(train_loader): 
                        self.optimizer.zero_grad()
                        batch_loss.backward()
                        self.optimizer.step()
                        batch_loss = 0
                    pbar.update(1)
                            
            
            train_loss = epoch_loss / self.n_train
            val_loss = self.eval_network(val_loader)
            self.scheduler.step(val_loss)
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model_params = self.model.state_dict()
            print('Train: {0:.6e} Val: {1:.6e} Best: {2:.6e} + LR: {3:0.2e}'.format(train_loss, val_loss, self.best_score, self.optimizer.param_groups[0]['lr']))
            df_entry = pd.Series({'epoch': self.last_epoch, 'train loss': train_loss, 'validation loss': val_loss})
            self.training_history = self.training_history.append(df_entry, ignore_index=True)
            self.last_epoch += 1
            if self.early_stopping.step(val_loss):
                print('EARLY STOPPING')
                done = True
            self.save_model()
            
                
                
        print('END TRAINING')
        
    def generate_performance_metrics(self, file_name, model_type, metrics=['log_flux', 'flux', 'MSE', 'SMSE', 'MAE', 'ME', 'PE'], remake_preds=False, remake_metrics=False):
        series_file = self.save_folder / file_name
        self.dataset.set_transform(self.transform_eval)
        n_all = len(self.dataset)
        data_types = ['all', 'train', 'valid', 'test']
        
        if remake_preds or not series_file.exists():
            sample = self.dataset[0]
            img_size = (sample[self.in_feature].shape[1], sample[self.in_feature].shape[2])
            #out_size = (sample[self.out_feature].shape[1], sample[self.out_feature].shape[2])
            data_shape = (n_all, img_size[0], img_size[1])
            with h5py.File(series_file, 'w') as hf:
                for out_type in self.out_types:
                    hf.create_dataset(model_type + '_' + out_type, shape=data_shape, dtype=np.float32)
                hf.create_dataset('mask', shape=data_shape)
                hf.create_dataset('type', shape=(n_all, 1), dtype=h5py.string_dtype())
            with h5py.File(series_file, 'a') as hf:
                data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, pin_memory=True)
                with tqdm(range(n_all), desc='Calculating model prections') as pbar:
                    idx = 0
                    for batch in data_loader:
                        in_imgs = np.array(batch[self.in_feature])
                        out_imgs = np.array(batch[self.out_feature])
                        in_img_t = batch[self.in_feature]
                        in_img_t = in_img_t.to(self.device)
            
                        mask_data_type, mask_idx = self.get_mask_loc()
                        if mask_data_type == self.in_feature:
                            mask = in_imgs[:, mask_idx:(mask_idx+1), :, :] != in_imgs[0, mask_idx, 0, 0]
                        else:
                            mask = out_imgs[:, mask_idx:(mask_idx+1), :, :] != in_imgs[0, mask_idx, 0, 0]
                        
                        hf['mask'][idx:idx+1] = mask
                    
                        with torch.no_grad():
                            out_pred = self.model(in_img_t)

                        out_pred = np.array(out_pred.cpu())
                        
                        for j in range(len(self.out_types)):
                            out_imgs[:, j, :, :] = out_imgs[:, j, :, :] * mask 
                            out_pred[:, j, :, :] = out_pred[:, j, :, :] * mask 

                        if idx in self.train_idx:
                            data_type = 'train'
                        elif idx in self.val_idx:
                            data_type = 'valid'
                        elif idx in self.test_idx:
                            data_type = 'test'
                        
                        for j in range(len(self.out_types)):
                            hf[model_type + '_' + self.out_types[j]][idx] = out_pred[0, j]
                        hf['type'][idx] = data_type
                        idx += 1
                        pbar.update(1)
                    types = hf['type'][:].reshape(-1)
                    for data_type in data_types:
                        if data_type == 'all':
                            hf['type'].attrs[data_type] = n_all
                        else:
                            hf['type'].attrs[data_type] = np.sum(data_type == types)
        
        if remake_metrics:
            
            temp_series_file = self.save_folder / 'temp_series.h5'
            
            list_metrics = ['log_flux', 'flux']
            image_metrics = ['MSE', 'SMSE', 'MAE', 'ME', 'PE']
            agg_types = ['mean', 'median']
            self.dataset.set_transform(None)
            
            sample = self.dataset[0]
            img_size = (sample[self.in_feature].shape[1], sample[self.in_feature].shape[2])
            
            if temp_series_file.exists():
                os.remove(temp_series_file)
                
            with h5py.File(series_file, 'a') as hf:
                for data_type in data_types:
                    n_type = hf['type'].attrs[data_type]
                    for metric in metrics:
                        for out_type in self.out_types:
                            metric_name = data_type + '/' + metric + '/' + model_type + '/' + out_type

                            if remake_metrics:
                                if metric_name in hf:
                                    del hf[metric_name]

                            if metric in image_metrics:
                                with h5py.File(temp_series_file, 'a') as hf_temp:
                                    hf_temp.create_dataset(metric_name, shape=(n_type, img_size[0], img_size[1]))

                            if not metric_name in hf:
                                if metric in list_metrics:
                                    hf.create_dataset(metric_name, shape=(n_type,), chunks=True, dtype=np.float32)
                                if metric in image_metrics:
                                    hf.create_group(metric_name)
                                
                with h5py.File(temp_series_file, 'a') as hf_temp:
                    inverse_normal, inverse_log = self.transform_output_inverse
                    n_all = hf['type'].attrs['all']
                    
                    type_indices = {}
                    for data_type in data_types:
                        type_indices[data_type] = 0
                        
                    for i in tqdm(range(n_all), desc='Calculating model metrics'):
                        
                        mask = 1 - hf['mask'][i]
                        img_data_type = hf['type'][i][0]
                        true_imgs = {}
                        sample_out = self.dataset[i]['out']
                        for j in range(len(self.out_types)):
                            true_imgs[self.out_types[j]] = sample_out[j]
                            
                        if any(m in metrics for m in ['PE']):
                            inv_imgs = inverse_log(inverse_normal(sample_out))
                            true_inv_imgs = {}
                            for j in range(len(self.out_types)):
                                true_inv_imgs[self.out_types[j]] = inv_imgs[j]
                                
                        if any(m in metrics for m in ['log_flux', 'flux', 'emission', 'PE']):
                            inv_normal_imgs = {}
                            
                            inv_normal_all = inverse_normal(hf[model_type + '_' + out_type][i])
                            for j in range(len(self.out_types)):
                                inv_normal_imgs[self.out_types[j]] = inv_normal_all[j]
                        if any(m in metrics for m in ['flux', 'emission', 'PE']):
                            inv_log_imgs = {}
                            inv_log_all = inverse_log(inv_normal_all)
                            for j in range(len(self.out_types)):
                                inv_log_imgs[self.out_types[j]] = inv_log_all[j]
                        if any(m in metrics for m in ['MSE', 'SMSE', 'MAE', 'ME']):
                            diff_data = {}
                            for out_type in self.out_types:
                                diff_data[out_type] = hf[model_type + '_' + out_type][i] - true_imgs[out_type]
                        if any(m in metrics for m in ['PE']):
                            inverse_diff_data = {}
                            for out_type in self.out_types:
                                inverse_diff_data[out_type] = true_inv_imgs[out_type] - inv_log_imgs[out_type]
                                
                        for metric in metrics:
                            for data_type in data_types:
                                for out_type in self.out_types:
                                    n_type = hf['type'].attrs[data_type]
                                    metric_name = data_type + '/' + metric + '/' + model_type + '/' + out_type

                                    if data_type == img_data_type or data_type == 'all':
                                        if metric == 'log_flux':
                                            #print(mask.shape)
                                            #print(inv_normal_imgs[out_type].shape)
                                            hf[metric_name][type_indices[data_type]] = np.ma.masked_array(inv_normal_imgs[out_type], mask).sum()
                                        elif metric == 'flux':
                                            hf[metric_name][type_indices[data_type]] = np.ma.masked_array(inv_log_imgs[out_type], mask).sum()
                                        elif metric == 'MSE':
                                            hf_temp[metric_name][type_indices[data_type]] = np.square(diff_data[out_type])
                                        elif metric == 'SMSE':
                                            hf_temp[metric_name][type_indices[data_type]] = np.multiply(np.sign(diff_data[out_type]), np.square(diff_data[out_type]))
                                        elif metric == 'MAE':
                                            hf_temp[metric_name][type_indices[data_type]] = np.abs(diff_data[out_type])
                                        elif metric == 'ME':
                                            hf_temp[metric_name][type_indices[data_type]] = diff_data[out_type]
                                        elif metric == 'PE':
                                            data = 100 * np.divide(inverse_diff_data[out_type], true_inv_imgs[out_type])
                                            data[np.isnan(data)] = 100
                                            hf_temp[metric_name][type_indices[data_type]] = data
                                        elif metric == 'emission':
                                            hf_temp[metric_name][type_indices[data_type]] = np.ma.masked_array(inv_log_imgs[out_type], mask)
                                        #hf[metric_name].attrs['num_processed'] = type_indices[data_type] + 1
                        type_indices['all'] = type_indices['all'] + 1
                        type_indices[img_data_type] = type_indices[img_data_type] + 1
                        
                    for data_type in data_types:
                        n_type = hf['type'].attrs[data_type]
                        for metric in metrics:
                            if metric in image_metrics:
                                for out_type in self.out_types:
                                    metric_name = data_type + '/' + metric + '/' + model_type + '/' + out_type
                                    raw_img_data = hf_temp[metric_name][:]
                                    
                                    for agg_type in ['mean', 'median']:
                                        agg_metric_name = metric_name + '/' + agg_type
                                        
                                        if agg_type == 'mean':
                                            img_data = np.mean(raw_img_data, axis=0)
                                            list_data = np.mean(raw_img_data, axis=(1, 2))
                                        elif agg_type == 'median':
                                            img_data = np.median(raw_img_data, axis=0)
                                            list_data = np.median(raw_img_data, axis=(1, 2))
                                        hf.create_dataset(agg_metric_name + '/img', shape=img_size)
                                        hf.create_dataset(agg_metric_name + '/list', shape=(n_type,), chunks=True, dtype=np.float32)
                                        hf[agg_metric_name + '/img'][:] = img_data
                                        hf[agg_metric_name + '/list'][:] = list_data
                                        
            if temp_series_file.exists():
                os.remove(temp_series_file)
                    
        self.dataset.set_transform(self.transform_train)
               
    def generate_multiout_performance_images(self, series_name, image_folder, model_type, regenerate=False, metrics=['MSE', 'SMSE', 'MAE', 'ME', 'PE'], std_colorbar_range=4, dpi=None):
        series_file = self.save_folder / series_name
        image_save_folder = self.save_folder / image_folder
        image_save_folder.mkdir(parents=True, exist_ok=True)
        
        data_types = ['all', 'train', 'valid', 'test']
        agg_types = ['mean', 'median']
        
        with h5py.File(series_file, 'a') as hf:
            for metric in metrics:
                for data_type in data_types:
                    for out_type in self.out_types:
                        for agg_type in agg_types:

                            highest_std = 0
                            metric_name = data_type + '/' + metric + '/' + model_type + '/' + out_type + '/' + agg_type + '/img'
                            highest_std = max(np.abs(np.std(hf[metric_name][:])), highest_std)

                            if metric in ['MSE', 'MAE']:
                                colormap = 'Reds'
                                color_limits = (0, std_colorbar_range * highest_std)
                            elif metric in ['SMSE', 'ME', 'PE']:
                                colormap = 'seismic'
                                color_limits = (-std_colorbar_range * highest_std, std_colorbar_range * highest_std)

                            img_file_name = data_type + '_' + metric + '_' + model_type + '_' + out_type + '_' + agg_type + '.png'

                            if regenerate or not img_file_name.exists():

                                data = hf[metric_name][:]
                                if metric == 'PE':
                                    mean_val = np.mean(np.abs(data))
                                    std_val = np.std(np.abs(data))
                                else:
                                    mean_val = np.mean(data)
                                    std_val = np.std(data)
                                mean_str = 'Mean: {0:.2E}\nSTD: {1:.2E}'.format(mean_val, std_val)
                                mean_str_no_new_line = mean_str.replace('\n', ' ')

                                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
                                ax.axis('off')
                                ax.set_aspect('equal')
                                mesh = ax.pcolormesh(data, cmap=colormap)
                                mesh.set_clim(color_limits[0], color_limits[1])
                                plt.colorbar(mesh,ax=ax)

                                props = dict(boxstyle='square', facecolor='wheat', linewidth=0, alpha=0.15)
                                ax.text(0.05, 0.95, mean_str, transform=ax.transAxes, fontsize=8, va='top', ha='left', bbox=props)
                                fig.savefig(image_save_folder / img_file_name, dpi=dpi)
                                plt.close(fig)
                                
    def generate_multiout_comparison_images(self, series_name, image_folder, model_type, idx, std_scale_range=3, dpi=None):
        series_file = self.save_folder / series_name
        image_save_folder = self.save_folder / image_folder / 'comparisons'
        image_save_folder.mkdir(parents=True, exist_ok=True)
        
        for img_idx in idx:
        
            with h5py.File(series_file, 'a') as hf:

                mask = 1 - hf['mask'][img_idx]

                true_imgs = {}
                self.dataset.set_transform(self.transform_eval)
                sample_out = self.dataset[img_idx]['out']
                for j in range(len(self.out_types)):
                    true_imgs[self.out_types[j]] = np.ma.masked_array(sample_out[j].numpy(), mask)
                self.dataset.set_transform(self.transform_train)

                pred_imgs = {}
                for out_type in self.out_types:
                    pred_imgs[out_type] = np.ma.masked_array(hf[model_type + '_' + out_type][img_idx], mask)

            mean_centers = {}
            std_scales = {}
            for out_type in self.out_types:
                mean_centers[out_type] = np.mean(true_imgs[out_type])
                std_scales[out_type] = np.max((np.std(true_imgs[out_type]), np.std(pred_imgs[out_type])))



            scale = 4
            nrows = len(self.out_types)
            fig_tot, ax_tot = plt.subplots(nrows=nrows, ncols=2, squeeze=False, figsize=(2*scale, nrows*scale))

            for r in range(nrows):
                for c in range(2):

                    out_type = self.out_types[r]
                    color_limits = (-std_scales[out_type] * std_scale_range, std_scales[out_type] * std_scale_range)
                    if c == 0:
                        data = true_imgs[out_type]
                        name = 'True ' + out_type
                    if c == 1:
                        data = pred_imgs[out_type]
                        name = 'Pred ' + out_type

                    ax_tot[r, c].axis('off')
                    ax_tot[r, c].set_aspect('equal')
                    
                    mesh_tot = ax_tot[r, c].pcolormesh(data, cmap='viridis')
                    mesh_tot.set_clim(color_limits[0], color_limits[1])
                    ax_tot[r, c].text(0.5, 0.0, name, transform=ax_tot[r, c].transAxes, fontsize=12, va='top',ha='center')
            fig_tot.suptitle('True and Predicted images')
            plt.show()
            fig_tot.savefig(image_save_folder / (str(img_idx) + '.png'), dpi=dpi)
            plt.close(fig_tot)
            
    def generate_in_out_reversal_comparison_images(self, series_name_1, series_name_2, image_folder, series_1_model_type, series_2_model_type, idx, std_scale_range=3, dpi='figure', im_format='pdf'):
        #Series 1 should be generated by model comparison
        #Series 2 should be generated by sunmodel
        series_file_1 = self.save_folder / series_name_1
        series_file_2 = self.save_folder / series_name_2
        image_save_folder = self.save_folder / image_folder / 'comparisons'
        image_save_folder.mkdir(parents=True, exist_ok=True)
        
        image_names = {'ew': 'He I', '0304': 'EUV'}
        
        for img_idx in idx:
            print(img_idx)
            with h5py.File(series_file_1, 'a') as hf1:
                with h5py.File(series_file_2, 'a') as hf2:
                    mask = 1 - hf1['mask'][img_idx]
                    

                    true_imgs = {}
                    self.dataset.set_transform(self.transform_eval)
                    sample = self.dataset[img_idx]
                    sample_in = sample['in']
                    sample_out = sample['out']
                    for j in range(len(self.in_types)):
                        true_imgs[self.in_types[j]] = np.ma.masked_array(sample_in[j].numpy(), mask)
                    for j in range(len(self.out_types)):
                        true_imgs[self.out_types[j]] = np.ma.masked_array(sample_out[j].numpy(), mask)
                    self.dataset.set_transform(self.transform_train)

                    pred_imgs = {}
                    for in_type in self.in_types:
                        pred_imgs[in_type] = np.ma.masked_array(hf1[series_1_model_type][img_idx], mask)
                    for out_type in self.out_types:
                        pred_imgs[out_type] = np.ma.masked_array(hf2[series_2_model_type + '_' + out_type][img_idx], mask)
                        
            img_types = self.in_types + self.out_types
            mean_centers = {}
            std_scales = {}
            for img_type in img_types:
                mean_centers[img_type] = np.mean(true_imgs[img_type])
                std_scales[img_type] = np.max((np.std(true_imgs[img_type]), np.std(pred_imgs[img_type])))

            scale = 4
            nrows = 2
            fig_tot, ax_tot = plt.subplots(nrows=nrows, ncols=2, squeeze=False, figsize=(2*scale, nrows*scale))
            
            for img_type in img_types:
                print('MSE ' + img_type + ':')
                print(np.average(np.square(true_imgs[img_type] - pred_imgs[img_type])))

            for r in range(nrows):
                for c in range(2):
                    img_type = img_types[1-c]
                    if r == 0 and c == 0:
                        data = true_imgs[img_type]
                        name = 'True'
                    elif r == 0 and c == 1:
                        data = pred_imgs[img_type]
                        name = 'Predicted'
                    elif r == 1 and c == 0:
                        data = pred_imgs[img_type]
                        name = 'Predicted'
                    elif r == 1 and c == 1:
                        data = true_imgs[img_type]
                        name = 'True'
                    
                    
                    #name = image_names[img_type]
                    color_limits = (-std_scales[img_type] * std_scale_range, std_scales[img_type] * std_scale_range)

                    ax_tot[r, c].axis('off')
                    ax_tot[r, c].set_aspect('equal')

                    mesh_tot = ax_tot[r, c].pcolormesh(data, cmap='viridis', rasterized=True)
                    mesh_tot.set_clim(color_limits[0], color_limits[1])
                    ax_tot[r, c].text(0.5, 0.0, name, transform=ax_tot[r, c].transAxes, fontsize=16, va='top',ha='center')
                    #if r == 0:
                        #ax_tot[r, c].text(0.5, 1.0, column_name, transform=ax_tot[r, c].transAxes, fontsize=19, va='bottom',ha='center')
                        
                    if c == 1:
                        arrow_dir = [(1, 0.5), (0, 0.5)]
                        axes_order = [0,1]
                        if r == 1:
                            arrow_dir = arrow_dir[::-1]
                            axes_order = axes_order[::-1]
                        con = ConnectionPatch(xyA=arrow_dir[0], xyB=arrow_dir[1], coordsA="axes fraction", coordsB="axes fraction",
                                              axesA=ax_tot[r, axes_order[0]], axesB=ax_tot[r, axes_order[1]], color="red",
                                              arrowstyle='->', mutation_scale=30, lw=5.0)
                        ax_tot[r, c].add_artist(con)
            cax = fig_tot.add_axes([ax_tot[1, 0].get_position().x0, ax_tot[1, 0].get_position().y0-0.08, ax_tot[1, 1].get_position().x1-ax_tot[1, 0].get_position().x0, 0.03])
            fig_tot.subplots_adjust(hspace=0.15, wspace=0.2)   
            #print(ax_tot.ravel().tolist())
            #fig_tot.colorbar(mesh_tot, ax=ax_tot.ravel().tolist(), orientation='horizontal', shrink=0.75)
            fig_tot.colorbar(mesh_tot, cax=cax, orientation='horizontal')
            plt.show()
            fig_tot.savefig(image_save_folder / (str(img_idx) + '.' + im_format), dpi=dpi)
            plt.close(fig_tot)
            
    def generate_feature_target_prediction_difference_images(self, series_name, image_folder, model_name, regenerate=False, idx=None, dpi=None, std_colorbar_range=[1, 1, 1, 1], im_format='png', top_left=(0, 0), crop_size=None):
        names = ['He I', 'EUV 17.1 nm', 'Error', 'Prediction']
        features = [self.in_types[0], self.out_types[0], model_name]
        
        series_file = self.save_folder / series_name
        image_save_folder = self.save_folder / image_folder / 'comparisons'
        image_save_folder.mkdir(parents=True, exist_ok=True)
        
        for img_idx in idx:
            print(img_idx)
            with h5py.File(series_file, 'a') as hf:
                mask = 1 - hf['mask'][img_idx]
                
                images = {}
                self.dataset.set_transform(self.transform_eval)
                sample = self.dataset[img_idx]
                sample_in = sample['in']
                sample_out = sample['out']
                for j in range(len(self.in_types)):
                    images[self.in_types[j]] = np.ma.masked_array(sample_in[j].numpy(), mask)
                for j in range(len(self.out_types)):
                    images[self.out_types[j]] = np.ma.masked_array(sample_out[j].numpy(), mask)
                    images[model_name] = np.ma.masked_array(hf[model_name + '_' + self.out_types[j]][img_idx], mask)
                self.dataset.set_transform(self.transform_train)
            
            mean_vals = {}
            std_vals = {}
            for feature in features:
                if feature in [self.out_types[0], model_name]:
                    mean_vals[feature] = np.mean(images[self.out_types[0]])
                    std_vals[feature] = np.max((np.std(images[self.out_types[0]]), np.std(images[model_name])))
                else:
                    mean_vals[feature] = np.mean(images[feature])
                    std_vals[feature] = np.std(images[feature])
            mean_vals['error'] = np.mean(images[features[-1]] - images[features[1]])
            std_vals['error'] = np.std(images[features[-1]] - images[features[1]])
            
            print('MSE:')
            print((np.average(np.square(images[features[1]] - images[features[2]]))))

            
            image_file_name = 'image_' + str(img_idx) + '.' + im_format

            fig, ax = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(6, 6))
            i = 0
            text_boxes = []
            for r in range(2):
                for c in range(2):
                    #props = dict(boxstyle='round', facecolor=None, edgecolor=None, alpha=0)
                    name_idx = r*2+c
                    crange = std_colorbar_range[name_idx]
                    if r == 1 and c == 0:
                        data = images[features[-1]] - images[features[1]]
                        if crop_size is not None:
                            data = data[top_left[0]:top_left[0]+crop_size[0],
                                        top_left[1]:top_left[1]+crop_size[1]]
                        mesh_err = ax[r, c].pcolormesh(data, cmap='seismic', rasterized=True)
                        
                        mesh_err.set_clim(-std_vals['error']*crange, std_vals['error']*crange)
                    else:
                        data = images[features[i]]
                        if crop_size is not None:
                            data = data[top_left[0]:top_left[0]+crop_size[0],
                                        top_left[1]:top_left[1]+crop_size[1]]
                        mesh = ax[r, c].pcolormesh(data, cmap='viridis', rasterized=True)
                        std_val = std_vals[features[i]]
                        #std_val = np.std(data)
                        mesh.set_clim(-std_val*crange, std_val*crange)
                        i += 1

                    #text_box = ax[r, c].text(0.5, -0.03, names[name_idx], transform=ax[r, c].transAxes, fontsize=12, va='top',ha='center', bbox=props)
                    #text_boxes.append(text_box)
                    ax[r, c].axis('off')
                    ax[r, c].set(adjustable='box', aspect='equal')
                    text_box = ax[r, c].text(0.5, 0.0, names[name_idx], transform=ax[r, c].transAxes, fontsize=16, va='top',ha='center')
                    text_boxes.append(text_box)

            fig.tight_layout()
            #cax_err = fig.add_axes([ax[0, 1].get_position().x1+0.2,ax[1, 1].get_position().y0,0.05,ax[0, 1].get_position().y1-ax[1, 1].get_position().y0])
            #cax = fig.add_axes([ax[0, 1].get_position().x1+0.05,ax[1, 1].get_position().y0,0.05,ax[0, 1].get_position().y1-ax[1, 1].get_position().y0])
            cax_err = fig.add_axes([ax[1, 0].get_position().x0, ax[1, 0].get_position().y0-0.19, ax[1, 1].get_position().x1-ax[1, 0].get_position().x0, 0.03])
            cax = fig.add_axes([ax[1, 0].get_position().x0, ax[1, 0].get_position().y0-0.09, ax[1, 1].get_position().x1-ax[1, 0].get_position().x0, 0.03])

            #fig.colorbar(mesh_err, ax=ax.ravel().tolist())
            fig.colorbar(mesh_err, cax=cax_err, orientation='horizontal')
            fig.colorbar(mesh, cax=cax, orientation='horizontal')

            fig.savefig(image_save_folder / image_file_name, dpi=dpi, bbox_extra_artists=text_boxes + [cax_err, cax], bbox_inches='tight')
            #shutil.copy(str(file_name_type), str(file_name_all))
            plt.show()
            #plt.savefig(file_name_type, dpi=dpi)
            plt.close(fig)