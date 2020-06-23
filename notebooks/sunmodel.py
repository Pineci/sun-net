import pathlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from unet import UNet
from dataset import SunImageDataset

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
    settings_model(model)
        Specifies the model to be used and trained
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
    show_image_predictions(mode)
        Shows images side-by-side of input, output, and predicted output images of the model
    train_network(epochs, batch_size, num_evals_per_epoch)
        Train the network up to the specified number of epochs with the given batch_size
    '''
    
    in_feature = 'ew'
    out_feature = '0304'
    
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
        
        #Load model if specified
        if load_model:
            self.load_model()
            
        #Using MSE Loss
        self.criterion = nn.MSELoss()
        

    def settings_model(self, model='unet'):
        '''This function specifies the model to be used. Currently, the only available model is 'unet',
        though in the future this can be expanded
        
        Parameters
        ----------
        
        model : str, optional
            Identifier for the model type to be used (default = 'unet')
        
        '''
        self.model_type = model
        if model == 'unet':
            self.model = UNet(n_channels=1, bilinear=False)
        else:
            raise ValueError('SunModel: Received model type ' + str(model) + ', must be one of: ' + str(['unet']))
        self.model = self.model.to(self.device)
            
    def settings_optimizer(self, optimizer='AdamW', lr=0.001, weight_decay=1e-7, momentum=0.8):
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
    
    def create_train_val_sets(self, val_proportion=0.2, random_state=None):
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
        self.val_proportion = val_proportion
        self.random_state = random_state
        n_tot = len(self.dataset)
        n_val = int(n_tot * self.val_proportion)
        n_train = n_tot - n_val
        indices = list(range(n_tot))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[:n_train], indices[n_train:n_val+n_train]
        
        train_tensor_dict = self.dataset[train_idx]
        x_train_mean = torch.mean(train_tensor_dict[self.in_feature]).item()
        x_train_std = torch.std(train_tensor_dict[self.in_feature]).item()
        y_train_mean = torch.mean(train_tensor_dict[self.out_feature]).item()
        y_train_std = torch.std(train_tensor_dict[self.out_feature]).item()
        
        transform = {self.in_feature: transforms.Normalize(x_train_mean, x_train_std),
                     self.out_feature: transforms.Normalize(y_train_mean, y_train_std)}
        self.dataset.set_transform(transform=transform)
        train, val = Subset(self.dataset, indices=train_idx), Subset(self.dataset, indices=val_idx)
        self.train = train
        self.val = val
        
    def save_model(self):
        '''Saves the model state and the state of the optimizer and scheduler. Also saves
        the number of epochs that have been trained on as well as the model properties.
        This saves all necessary information to reload the model and resume training
        from only the saved file specified by self.save_path (made in the constructor).
        
        '''
        model_properties = {}
        model_properties['model_type'] = self.model_type
        model_properties['optimizer_type'] = self.optimizer_type
        model_properties['scheduler_type'] = self.scheduler_type
        model_properties['val_proportion'] = self.val_proportion
        model_properties['random_state'] = self.random_state
        
        checkpoint = {}
        checkpoint['epoch'] = self.last_epoch
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['scheduler'] = self.scheduler.state_dict()
        checkpoint['model_properties'] = model_properties
        torch.save(checkpoint, self.save_path)
        
    def load_model(self):
        '''Loads the model from the specified file from self.save_path (made in the constructor)
        All states are loaded such that training may resume from where it was left off.
        
        '''
        checkpoint = torch.load(self.save_path)
        
        model_properties = checkpoint['model_properties']
        self.settings_model(model=model_properties['model_type'])
        self.settings_optimizer(optimizer=model_properties['optimizer_type'])
        self.settings_scheduler(scheduler=model_properties['scheduler_type'])
        self.create_train_val_sets(val_proportion=model_properties['val_proportion'], random_state=model_properties['random_state'])
        
        self.last_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        
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
        n_val = len(self.val)
        for batch in data_loader:
            in_imgs = batch[self.in_feature]
            out_imgs = batch[self.out_feature]
            in_imgs = in_imgs.to(self.device)
            out_imgs = out_imgs.to(self.device)
            
            with torch.no_grad():
                out_pred = self.model(in_imgs)
            loss = self.criterion(out_pred, out_imgs)
            total_loss += loss.item()
        self.model.train()
        return total_loss / n_val
    
    def show_image_predictions(self, mode='all'):
        '''A useful visualization for the outputs of the model.
        Plots 3 images for each instance in the desired dataset, the leftmost
        image being the input image, the middle image being the output, and rightmost
        being the model's predicted output image.
        
        Parameters
        ----------
        
        mode : str, optional
            Specifies which data to be visualized. Options include 'train' for training
            data, 'val' for validation data, or 'all' for both
        '''
        self.model.eval()
        if mode == 'train':
            data_loader = DataLoader(self.train, batch_size=1, shuffle=True, pin_memory=True)
        elif mode == 'val':
            data_loader = DataLoader(self.val, batch_size=1, shuffle=False, pin_memory=True)
        elif mode == 'all':
            data_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, pin_memory=True) 
        
        columns = 3
        rows = 1
        
        print('Left to Right: Input, Output, Predicted')
        
        for batch in data_loader:
            in_img = np.array(batch[self.in_feature])
            out_img = np.array(batch[self.out_feature])
            in_img_t = batch[self.in_feature]
            in_img_t = in_img_t.to(self.device)
            
            with torch.no_grad():
                out_pred = self.model(in_img_t)
            
            out_pred = np.array(out_pred.cpu())
        
            fig=plt.figure(figsize=(8, 8))
            fig.add_subplot(rows, columns, 1)
            plt.imshow(in_img.reshape(512, 512))
            fig.add_subplot(rows, columns, 2)
            plt.imshow(out_img.reshape(512, 512))
            fig.add_subplot(rows, columns, 3)
            plt.imshow(out_pred.reshape(512, 512))
            plt.show()
        self.model.train()
        
            
    def train_network(self, epochs=10, batch_size=4):
        '''Trains the model using the specified number of epochs and batch size. Uses
        the preconfigured model, optimizer, scheduler, and train/validation sets. 
        
        Parameters
        ----------
        
        epochs : int, optional
            The number of epochs to train for
        batch_size : int, optional
            The number of instances in every batch
        '''
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(self.val, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        writer = SummaryWriter(comment='UNet')
        n_train = len(train_loader)
        
        while self.last_epoch < epochs:
            print('EPOCH: ' + str(self.last_epoch))
            self.model.train()
            epoch_loss = 0
            for batch in train_loader:
                in_imgs = batch[self.in_feature]
                out_imgs = batch[self.out_feature]
                in_imgs = in_imgs.to(self.device)
                out_imgs = out_imgs.to(self.device)
                
                out_pred = self.model(in_imgs)
                loss = self.criterion(out_pred, out_imgs)
                epoch_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                    
                    
            
            train_loss = epoch_loss / n_train
            val_loss = self.eval_network(val_loader)
            writer.add_scalar('Epoch Loss/train', train_loss, self.last_epoch)
            writer.add_scalar('Epoch Loss/validation', val_loss, self.last_epoch)
            self.scheduler.step(val_loss)
            print('Train: ' + str(train_loss) + ' Val: ' + str(val_loss) + ' LR: ' + str(self.optimizer.param_groups[0]['lr']))
            self.last_epoch += 1
            self.save_model()
                
        writer.close()
               