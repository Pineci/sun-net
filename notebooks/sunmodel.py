import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
import pandas as pd

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
    
    in_feature = 'ew'
    out_feature = '0304'
    
    power_transforms = (None, 0.0)
    
    def __init__(self, device='cpu', save_folder='models', save_name='model.pt', load_model=False):
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
        self.save_path = pathlib.Path(save_folder)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_path / save_name
        
        self.last_epoch = 0
        self.training_history = pd.DataFrame()
        
        #Set up device
        self.device_type = device
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device == 'gpu':
            self.device = torch.device('cuda:0')
        else:
            raise ValueError('SunModel: Received device type ' + str(device) + ', must be one of: ' + str(['cpu', 'gpu']))
            
        #Set up input data frame
        self.dataset = SunImageDataset()
        
        self.loaded_model = False
        #Load model if specified
        if load_model and self.save_path.exists():
            self.load_model()
            self.loaded_model = True
        
        

    def settings_model(self, model='unet', add_distance_channel=False, add_latitude_channel=False, dropout=None):
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
        
        n_channels = 1
        if self.add_distance_channel:
            n_channels += 1
        if self.add_latitude_channel:
            n_channels += 1
            
        if model == 'unet':
            self.model = UNet(n_channels=n_channels, bilinear=False, dropout=dropout)
        elif model == 'fcn8':
            self.model = FCN('8', n_channels=n_channels, single_pixel_pred=False)
        elif model == 'fcn16':
            self.model = FCN('16', n_channels=n_channels, single_pixel_pred=False)
        elif model == 'fcn32':
            self.model = FCN('32', n_channels=n_channels, single_pixel_pred=False)
        elif model == 'fcn8-single':
            self.model = FCN('8', n_channels=n_channels, single_pixel_pred=True)
        elif model == 'fcn16-single':
            self.model = FCN('16', n_channels=n_channels, single_pixel_pred=True)
        elif model == 'fcn32-single':
            self.model = FCN('32', n_channels=n_channels, single_pixel_pred=True)
        elif model == 'pixelwise':
            self.model = Pixelwise(n_channels=n_channels)
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
        
    def create_train_val_test_sets(self, val_proportion=0.2, test_proportion=0.2, random_state=None, power_transforms=(None, 0.0)):
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
        self.best_score = np.inf
        self.val_proportion = val_proportion
        self.test_proportion = test_proportion
        self.random_state = random_state
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
        train_idx, val_idx, test_idx = indices[:n_train], indices[n_train:n_val+n_train], indices[n_val+n_train:n_tot]
        
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        
        train_tensor_dict = self.dataset[train_idx]
        x_train_data = train_tensor_dict[self.in_feature]
        y_train_data = train_tensor_dict[self.out_feature]
        x_train_min = np.min(self.dataset[indices][self.in_feature])
        y_train_min = np.min(self.dataset[indices][self.out_feature])
        
        x_transform, y_transform = self.power_transforms
        if x_transform is not None:
            x_train_data = boxcox(x_train_data - x_train_min + 1, x_transform)
        if y_transform is not None:
            y_train_data = boxcox(y_train_data - y_train_min + 1, y_transform)
        
        x_train_mean = np.mean(x_train_data)
        x_train_std = np.std(x_train_data)
        y_train_mean = np.mean(y_train_data)
        y_train_std = np.std(y_train_data)
        
        features = [self.in_feature, self.out_feature]
        self.transform_train = tvtf.Compose(self.generate_transform(features, mins=[x_train_min, y_train_min],
                                                                    means=[x_train_mean, y_train_mean],
                                                                    stds=[x_train_std, y_train_std],
                                                                    add_rotation=True, 
                                                                    add_distance=self.add_distance_channel))
        self.transform_eval = tvtf.Compose(self.generate_transform(features, mins=[x_train_min, y_train_min],
                                                                   means=[x_train_mean, y_train_mean],
                                                                   stds=[x_train_std, y_train_std],
                                                                   add_rotation=False, 
                                                                   add_distance=self.add_distance_channel))
        self.transform_output_inverse = self.generate_transform(features, mins=[x_train_min, y_train_min],
                                                                means=[x_train_mean, y_train_mean],
                                                                stds=[x_train_std, y_train_std],
                                                                inverse=True)
        
        self.dataset.set_transform(transform=self.transform_train)
        train, val, test = Subset(self.dataset, indices=train_idx), Subset(self.dataset, indices=val_idx), Subset(self.dataset, indices=test_idx)
        self.train = train
        self.val = val
        self.test = test
        
    def generate_transform(self, features, mins=[0, 0], means=[0, 0], stds=[1, 1], add_rotation=True, add_distance=False, add_latitude=False, inverse=False):
        transform_list = []
        
        if not inverse:
            transform_list.append(transforms.PowerTransform(features, 
                                                        lambdas=self.power_transforms, 
                                                        mins=mins))
            
            if add_distance or add_latitude:
                means[0] = [means[0]]
                stds[0] = [stds[0]]
            

            if add_distance:
                transform_dist_channel = transforms.AddDistChannel(features)
                transform_list.append(transform_dist_channel)
                dist_mean, dist_std = transform_dist_channel.get_mean_std()
                means[0] = means[0] + [dist_mean]
                stds[0] = stds[0] + [dist_std]
            
            if add_latitude:
                transform_lat_channel = transforms.AddLatitudeChannel(features)
                transform_list.append(transform_lat_channel)
                lat_mean, lat_std = transform_lat_channel.get_mean_std()
                means[0] = means[0] + [lat_mean]
                stds[0] = stds[0] + [lat_std]
                
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
        
        '''
        settings_model(self, model='unet', add_distance_channel=False, dropout=None)
        settings_criterion(self, criterion='MSE')
        settings_optimizer(self, optimizer='AdamW', lr=0.001, weight_decay=1e-7, momentum=0.8, batch_size=4, num_steps_per_batch=1)
        settings_scheduler(self, scheduler='ReduceLROnPlateau', factor=0.1, patience=2, min_lr=1e-8)
        settings_early_stopping(self, patience=10, mode='min', min_delta=0, percentage=False)
        create_train_val_test_sets(self, val_proportion=0.2, test_proportion=0.2, random_state=None)
        
        model_properties = {}
        model_properties['model_type'] = self.model_type
        model_properties['optimizer_type'] = self.optimizer_type
        model_properties['scheduler_type'] = self.scheduler_type
        model_properties['criterion_type'] = self.criterion_type
        model_properties['val_proportion'] = self.val_proportion
        model_properties['test_proportion'] = self.test_proportion
        model_properties['random_state'] = self.random_state
        model_properties['add_distance_channel'] = self.add_distance_channel
        model_properties['dropout'] = self.dropout
        model_properties['batch_size'] = self.batch_size
        model_properties['num_steps_per_batch'] = self.num_steps_per_batch
        
        model_properties['early_stopping_mode'] = self.early_stopping_mode
        model_properties['early_stopping_patience'] = self.early_stopping_patience
        model_properties['early_stopping_percentage'] = self.early_stopping_percentage
        model_properties['early_stopping_min_delta'] = self.early_stopping_min_delta
        model_properties['early_stopping_state'] = self.early_stopping.get_state()
        
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
        '''
        
        model_properties = {}
        model_properties['model_type'] = self.model_type
        model_properties['add_distance_channel'] = self.add_distance_channel
        model_properties['add_latitude_channel'] = self.add_latitude_channel
        model_properties['dropout'] = self.dropout
        
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
        checkpoint = torch.load(self.save_path)
        
        '''
        model_properties = checkpoint['model_properties']
        self.settings_model(model=model_properties['model_type'], 
                            add_distance_channel=model_properties['add_distance_channel'],
                            dropout=model_properties['dropout'])
        self.settings_optimizer(optimizer=model_properties['optimizer_type'])
        self.settings_scheduler(scheduler=model_properties['scheduler_type'])
        if 'criterion_type' in model_properties.keys():
            self.settings_criterion(criterion=model_properties['criterion_type'])
        else:
            self.settings_criterion(criterion='MSE')
        if 'batch_size' in model_properties.keys():
            
        self.settings_early_stopping(mode=model_properties['early_stopping_mode'], min_delta=model_properties['early_stopping_min_delta'], patience=model_properties['early_stopping_patience'], percentage=model_properties['early_stopping_percentage'])
        self.early_stopping.load_state(model_properties['early_stopping_state'])
        self.create_train_val_test_sets(val_proportion=model_properties['val_proportion'], test_proportion=model_properties['test_proportion'], random_state=model_properties['random_state'])
        
       
        self.training_history = checkpoint['training_history']
        self.last_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_model_params = checkpoint['best_model']
        self.best_score = checkpoint['best_score']
        
        settings_model(self, model='unet', add_distance_channel=False, dropout=None)
        settings_criterion(self, criterion='MSE')
        settings_optimizer(self, optimizer='AdamW', lr=0.001, weight_decay=1e-7, momentum=0.8, batch_size=4, num_steps_per_batch=1)
        settings_scheduler(self, scheduler='ReduceLROnPlateau', factor=0.1, patience=2, min_lr=1e-8)
        settings_early_stopping(self, patience=10, mode='min', min_delta=0, percentage=False)
        create_train_val_test_sets(self, val_proportion=0.2, test_proportion=0.2, random_state=None)
        '''
        model_properties = checkpoint['model_properties']
        self.settings_model(model=model_properties['model_type'], 
                            add_distance_channel=model_properties['add_distance_channel'],
                            add_latitude_channel=model_properties['add_latitude_channel'],
                            dropout=model_properties['dropout'])
        
        self.settings_optimizer(optimizer=model_properties['optimizer_type'],
                                lr=model_properties['optimizer_lr'],
                                weight_decay=model_properties['optimizer_weight_decay'],
                                momentum=model_properties['optimizer_momentum'],
                                batch_size=model_properties['batch_size'],
                                num_steps_per_batch=model_properties['num_steps_per_batch'])
        
        self.settings_scheduler(scheduler=model_properties['scheduler_type'],
                                factor=model_properties['scheduler_factor'],
                                patience=model_properties['scheduler_patience'],
                                min_lr=model_properties['scheduler_min_lr'])
        
        self.settings_criterion(criterion=model_properties['criterion_type'])
            
        self.settings_early_stopping(mode=model_properties['early_stopping_mode'],
                                     min_delta=model_properties['early_stopping_min_delta'], 
                                     patience=model_properties['early_stopping_patience'],
                                     percentage=model_properties['early_stopping_percentage'])
        self.early_stopping.load_state(model_properties['early_stopping_state'])
        
        self.create_train_val_test_sets(val_proportion=model_properties['val_proportion'], 
                                        test_proportion=model_properties['test_proportion'], 
                                        random_state=model_properties['random_state'],
                                        power_transforms=model_properties['power_transforms'])
        
       
        self.training_history = checkpoint['training_history']
        self.last_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_model_params = checkpoint['best_model']
        self.best_score = checkpoint['best_score']
        
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
            
            mask = in_imgs[:, 0:1, :, :] != in_imgs[0, 0, 0, 0]
            
            with torch.no_grad():
                out_pred = self.model(in_imgs)
            loss = self.criterion(out_pred * mask, out_imgs * mask) / torch.sum(mask)
            total_loss += loss.item()
        self.model.train()
        return total_loss / n
    
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
            out_img = np.array(batch[self.out_feature])
            in_img_t = batch[self.in_feature]
            in_img_t = in_img_t.to(self.device)
            
            with torch.no_grad():
                out_pred = self.model(in_img_t)
            
            out_pred = np.array(out_pred.cpu())
        
            fig=plt.figure(figsize=(8, 8))
            
            if self.add_distance_channel and False:
                ax = fig.add_subplot(rows, columns, dist_idx)
                ax.set_yticklabels([])
                #ax.set_xticklabels([])
                plt.imshow(dist_img.reshape(512, 512))
                
            ax = fig.add_subplot(rows, columns, in_idx)
            ax.set_yticklabels([])
            #ax.set_xticklabels([])
            plt.imshow(in_img.reshape(512, 512))
            plt.clim(color_limits[0][0], color_limits[0][1])
            
            ax = fig.add_subplot(rows, columns, out_idx)
            ax.set_yticklabels([])
            #ax.set_xticklabels([])
            plt.imshow(out_img.reshape(512, 512))
            plt.clim(color_limits[1][0], color_limits[1][1])
            
            ax = fig.add_subplot(rows, columns, out_pred_idx)
            ax.set_yticklabels([])
            #ax.set_xticklabels([])
            plt.imshow(out_pred.reshape(512, 512))
            plt.clim(color_limits[1][0], color_limits[1][1])
            plt.show()
        self.model.train()
        self.dataset.set_transform(transform=self.transform_train)
                               
    def get_image_prediction(self, mode='all', include_original=True):
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
        elif mode == 'val':
            data_loader = DataLoader(self.val, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'test':
            data_loader = DataLoader(self.test, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'all':
            data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, pin_memory=True)
            
        image_df = pd.DataFrame()
        
        idx = 0
        for batch in data_loader:
            in_img = np.array(batch[self.in_feature])
            out_img = np.array(batch[self.out_feature])
            in_img_t = batch[self.in_feature]
            in_img_t = in_img_t.to(self.device)
            
            mask = in_img[:, 0:1, :, :] != in_img[0, 0, 0, 0]
            
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
                
            
            if include_original:
                new_entry = pd.Series({'idx': idx,
                                       'type': data_type,
                                       self.in_feature: in_img.reshape(512, 512), 
                                       self.out_feature: out_img.reshape(512, 512), 
                                       'predicted': out_pred.reshape(512, 512)})
            else:
                new_entry = pd.Series({'idx': idx, 
                                       'type': data_type,
                                       'predicted': out_pred.reshape(512, 512)})
            image_df = image_df.append(new_entry, ignore_index=True)
            idx += 1
            
        self.model.train()
        self.dataset.set_transform(transform=self.transform_train)
        return image_df
    
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
        
        done = self.early_stopping.step(None, check=True)
        while self.last_epoch < epochs and not done:
            print('EPOCH: ' + str(self.last_epoch))
            self.model.train()
            epoch_loss, batch_loss, num_steps = 0, 0, 0
            for batch in train_loader:
                in_imgs = batch[self.in_feature]
                out_imgs = batch[self.out_feature]
                in_imgs = in_imgs.to(self.device)
                out_imgs = out_imgs.to(self.device)
                
                mask = in_imgs[:, 0:1, :, :] != in_imgs[0, 0, 0, 0]
                
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
                            
            
            train_loss = epoch_loss / self.n_train
            val_loss = self.eval_network(val_loader)
            self.scheduler.step(val_loss)
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model_params = self.model.state_dict()
            print('Train: ' + str(train_loss) + ' Val: ' + str(val_loss) + ' Best: ' + str(self.best_score) + ' LR: ' + str(self.optimizer.param_groups[0]['lr']))
            df_entry = pd.Series({'epoch': self.last_epoch, 'train loss': train_loss, 'validation loss': val_loss})
            self.training_history = self.training_history.append(df_entry, ignore_index=True)
            self.last_epoch += 1
            if self.early_stopping.step(val_loss):
                print('EARLY STOPPING')
                done = True
            self.save_model()
            
                
                
        print('END TRAINING')
               