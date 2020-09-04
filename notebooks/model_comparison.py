import pathlib
import numpy as np
import pandas as pd
import h5py
import os
import shutil
import gc
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import fsolve


import torch
from torch.utils.data import DataLoader

from sunmodel import SunModel
from dataset import SunImageDataset
import utils

class ModelComparison(object):
    
    in_feature = 'in'
    out_feature = 'out'
    
    data_types = ['all', 'train', 'valid', 'test']
    
    img_size = (864, 864)
    
    def __init__(self, root_dir='models', folder='run_1', load_settings=True):
        self.save_folder = pathlib.Path(root_dir + '/' + folder)
        self.comparison_folder = self.save_folder / 'comparisons'
        self.aggregate_folder = self.save_folder / 'aggregate'
        self.aggregate_data_types_folders = {}
        self.comparison_data_types_folders = {}
        for data_type in self.data_types:
            self.aggregate_data_types_folders[data_type] = self.aggregate_folder / data_type
            self.comparison_data_types_folders[data_type] = self.comparison_folder / data_type
        
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.comparison_folder.mkdir(parents=True, exist_ok=True)
        self.aggregate_folder.mkdir(parents=True, exist_ok=True)
        for data_type in self.data_types:
            self.aggregate_data_types_folders[data_type].mkdir(parents=True, exist_ok=True)
            self.comparison_data_types_folders[data_type].mkdir(parents=True, exist_ok=True)
            
        self.settings_file = self.save_folder / 'model_settings.pt'
        self.series_file = self.save_folder / 'series.h5'
        self.temp_series_file = self.save_folder / 'temp.h5'
        
        self.settings_criterion()
        self.settings_optimizer()
        self.settings_scheduler()
        self.settings_early_stopping()
        self.settings_validation_test()
        self.settings_models()
        self.settings_train()
        
        if load_settings and self.settings_file.exists():
            self.load_settings()
        
    def settings_criterion(self, criterion='MSE'):
        self.criterion_type = criterion
        
    def settings_optimizer(self, optimizer='AdamW', lr=1e-3, weight_decay=1e-7, momentum=0.8, ):
        self.optimizer_type = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        
    def settings_scheduler(self, scheduler='ReduceLROnPlateau', factor=0.2, patience=5, min_lr=1e-8):
        self.scheduler_type = scheduler
        self.scheduler_factor = factor
        self.patience = patience
        self.min_lr = min_lr
        
    def settings_early_stopping(self, patience=10, mode='min', min_delta=0, percentage=False):
        self.early_stopping_patience = patience
        self.early_stopping_mode = mode
        self.early_stopping_min_delta = min_delta
        self.early_stopping_percentage = percentage
        
    def settings_validation_test(self, val_proportion=0.2, test_proportion=0.2, random_state=None, power_transforms=(None, 0.0)):
        self.val_proportion = val_proportion
        self.test_proportion = test_proportion
        self.power_transforms = power_transforms
        if random_state is None:
            self.random_state = np.random.randint(0, 1e9, size=1)
        else:
            self.random_state = random_state
            
    def settings_models(self, model_types=['unet', 'fcn8', 'fcn16', 'fcn32'], dropout=None, use_batch_norm=False):
        self.model_types = model_types
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
    def settings_train(self, epochs=100, batch_size=4, num_steps_per_batch=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_steps_per_batch = num_steps_per_batch
        
    def save_settings(self):
        settings = {}
        
        settings['criterion_type'] = self.criterion_type
        
        settings['optimizer_type'] = self.optimizer_type
        settings['lr'] = self.lr
        settings['weight_decay'] = self.weight_decay
        settings['momentum'] = self.momentum
        
        settings['scheduler_type'] = self.scheduler_type
        settings['scheduler_factor'] = self.scheduler_factor
        settings['patience'] = self.patience
        settings['min_lr'] = self.min_lr
        
        settings['early_stopping_patience'] = self.early_stopping_patience
        settings['early_stopping_mode'] = self.early_stopping_mode
        settings['early_stopping_min_delta'] = self.early_stopping_min_delta
        settings['early_stopping_percentage'] = self.early_stopping_percentage
        
        settings['val_proportion'] = self.val_proportion
        settings['test_proportion'] = self.test_proportion
        settings['random_state'] = self.random_state
        settings['power_transforms'] = self.power_transforms
       
        settings['model_types'] = self.model_types
        settings['dropout'] = self.dropout
        settings['use_batch_norm'] = self.use_batch_norm
        
        settings['epochs'] = self.epochs
        settings['batch_size'] = self.batch_size
        settings['num_steps_per_batch'] = self.num_steps_per_batch
        
        torch.save(settings, self.settings_file)
        
    def load_settings(self):
        settings = torch.load(self.settings_file)
        
        if 'use_batch_norm' not in settings.keys():
            settings['use_batch_norm'] = True
        
        self.settings_criterion(criterion=settings['criterion_type'])
        self.settings_optimizer(optimizer=settings['optimizer_type'],
                                lr=settings['lr'], 
                                weight_decay=settings['weight_decay'],
                                momentum=settings['momentum'])
        self.settings_scheduler(scheduler=settings['scheduler_type'],
                                factor=settings['scheduler_factor'],
                                patience=settings['patience'],
                                min_lr=settings['min_lr'])
        self.settings_early_stopping(patience=settings['early_stopping_patience'],
                                     mode=settings['early_stopping_mode'],
                                     min_delta=settings['early_stopping_min_delta'],
                                     percentage=settings['early_stopping_percentage'])
        self.settings_validation_test(val_proportion=settings['val_proportion'],
                                      test_proportion=settings['test_proportion'],
                                      random_state=settings['random_state'],
                                      power_transforms=settings['power_transforms'])
        self.settings_models(model_types=settings['model_types'],
                            dropout=settings['dropout'],
                            use_batch_norm=settings['use_batch_norm'])
        self.settings_train(epochs=settings['epochs'],
                            batch_size=settings['batch_size'],
                            num_steps_per_batch=settings['num_steps_per_batch'])
        
    def get_model_save_name(self, model_type, dist_channel, lat_channel):
        save_name = model_type
        if dist_channel:
            save_name = save_name + '_dist'
            if lat_channel:
                save_name = save_name + '_lat'
        return save_name
        
    def get_model_prediction_save_name(self, model_type, dist_channel, lat_channel):
        save_name = self.get_model_save_name(model_type, dist_channel, lat_channel)
        return save_name + '_predictions.h5'
    
    def get_entry_save_name(self, idx):
        return 'image_' + utils.pad_string(str(idx), pad='0', length=4) + '.pkl'
    
    def get_aggregate_save_name(self, model_type, metric):
        return model_type + '_' + metric + '.png'
    
    def get_all_model_types(self):
        all_model_types = []
        for dist_channel, lat_channel in zip([False, True, True], [False, False, True]):
            for model_type in self.model_types:
                all_model_types.append(self.get_model_save_name(model_type, dist_channel, lat_channel))
        return all_model_types
    
    def get_all_model_names(self):
        all_model_names = []
        for dist_channel, lat_channel in zip([False, True, True], [False, False, True]):
            for model_type in self.model_types:
                if model_type == 'fcn8':
                    save_name = 'FCN'
                elif model_type == 'unet':
                    save_name = 'U-Net'
                elif model_type == 'pixelwise':
                    save_name = 'Pixelwise'
                
                if dist_channel:
                    save_name = save_name + ' with'
                    save_name = save_name + ' Limb'
                    if lat_channel:
                        save_name = save_name + ' and Lat'
                #else:
                #    save_name = save_name + ' No Additional'
                #save_name = save_name + ' Features'
                all_model_names.append(save_name)
        return all_model_names
        
    def load_model(self, model_type, dist_channel, lat_channel, device='gpu'):
        model_save_name = self.get_model_save_name(model_type, dist_channel, lat_channel) + '.pt'
        model = SunModel(device=device, save_folder=self.save_folder, save_name=model_save_name, load_model=True)
        if not model.loaded_model:
            model.settings_dataset()
            model.settings_model(model=model_type, 
                                 add_distance_channel=dist_channel,
                                 add_latitude_channel=lat_channel,
                                 dropout=self.dropout,
                                 use_batch_norm=self.use_batch_norm)
            model.settings_optimizer(optimizer=self.optimizer_type, 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay, 
                                     momentum=self.momentum)
            model.settings_scheduler(scheduler=self.scheduler_type, 
                                     factor=self.scheduler_factor,
                                     patience=self.patience, 
                                     min_lr=self.min_lr)
            model.settings_early_stopping(mode=self.early_stopping_mode, 
                                          patience=self.early_stopping_patience,
                                          min_delta=self.early_stopping_min_delta, 
                                          percentage=self.early_stopping_percentage)
            model.settings_criterion(criterion=self.criterion_type)
            model.create_train_val_test_sets(val_proportion=self.val_proportion, 
                                             test_proportion=self.test_proportion, 
                                             random_state=self.random_state,
                                             power_transforms=self.power_transforms)
        return model
    
    def load_empty_model(self, add_distance_channel=False, add_latitude_channel=False):
        model = SunModel(device='cpu', load_model=False)
        model.settings_dataset()
        model.settings_model(add_distance_channel=add_distance_channel, add_latitude_channel=add_latitude_channel)
        model.settings_optimizer()
        model.settings_scheduler()
        model.settings_criterion()
        model.create_train_val_test_sets(val_proportion=self.val_proportion,
                                         test_proportion=self.test_proportion,
                                         random_state=self.random_state,
                                         power_transforms=self.power_transforms)
        return model
        
    
    def train_models(self):
        self.save_settings()
        for model_type in self.model_types:
            for dist_channel, lat_channel in zip([False, True, True], [False, False, True]):
                
                print('Training model type ' + self.get_model_save_name(model_type, dist_channel, lat_channel))
                
                model = self.load_model(model_type, dist_channel, lat_channel)
                model.train_network(epochs=self.epochs)
                
                hf_save_file = self.save_folder / self.get_model_prediction_save_name(model_type, dist_channel, lat_channel)
                if not hf_save_file.exists():
                    model.get_image_prediction(hf_save_file, mode='all', include_original=False)
                    #df = model.get_image_prediction(hf_save_file, mode='all', include_original=False)
                    #df.to_pickle(df_save_file)
                    #del df
                del model
                gc.collect()
                
    def generate_baseline_series(self, remake=False):
        n_all = len(SunImageDataset())
        
        if remake or not self.series_file.exists():
            print('Constructing new baseline series ...', flush=True)
            data_shape = (n_all, self.img_size[0], self.img_size[1])
            with h5py.File(self.series_file, 'w') as hf:
                hf.create_dataset(self.in_feature, shape=data_shape, dtype=np.float32, chunks=True)
                hf.create_dataset(self.out_feature, shape=data_shape, dtype=np.float32, chunks=True)
                hf.create_dataset('mask', shape=data_shape, dtype=np.float32, chunks=True)
                hf.create_dataset('date', shape=(n_all,))
                hf[self.in_feature].attrs['num_processed'] = 0
                hf[self.out_feature].attrs['num_processed'] = 0
                hf['mask'].attrs['num_processed'] = 0
                hf['date'].attrs['num_processed'] = 0
        
        with h5py.File(self.series_file, 'a') as hf:
            num_processed = min(hf[self.in_feature].attrs['num_processed'],
                                hf[self.out_feature].attrs['num_processed'])
            if num_processed < n_all:
                empty_model = self.load_empty_model()
                empty_model.dataset.set_transform(transform=empty_model.transform_eval)
                dataset = empty_model.dataset
                for i in tqdm(range(num_processed, n_all), desc='Adding baseline images'):
                    sample_dict = dataset[i]
                    in_img = sample_dict[self.in_feature]
                    out_img = sample_dict[self.out_feature]

                    mask = in_img[0:1, :, :] != in_img[0, 0, 0]
                    in_img = in_img * mask
                    out_img = out_img * mask

                    hf[self.in_feature][i:i+1] = in_img
                    hf[self.out_feature][i:i+1] = out_img
                    hf['mask'][i:i+1] = mask
                    hf['date'][i] = dataset.get_date(i)

                    hf['date'].attrs['num_processed'] = i+1
                    hf['mask'].attrs['num_processed'] = i+1
                    hf[self.in_feature].attrs['num_processed'] = i+1
                    hf[self.out_feature].attrs['num_processed'] = i+1
                
                del empty_model
        
    def generate_prediction_series(self, remake=False):
        for model_type in self.model_types:
            for dist_channel, lat_channel in zip([False, True, True], [False, False, True]):
                hf_filename = self.save_folder / self.get_model_prediction_save_name(model_type, dist_channel, lat_channel)
                if hf_filename.exists():
                    feature_name = self.get_model_save_name(model_type, dist_channel, lat_channel)
                    
                    with h5py.File(self.series_file, 'a') as hf:
                        n = hf[self.in_feature].shape[0]
                        
                        headers_exist = feature_name in hf
                        type_exist = 'type' in hf
                        if remake and feature_name in hf:
                            del hf[feature_name]
                        if remake or not headers_exist:
                            data_shape = (n, self.img_size[0], self.img_size[1])
                            hf.create_dataset(feature_name, shape=data_shape, dtype=np.float32, chunks=True)
                            hf[feature_name].attrs['num_processed'] = 0
                        if not type_exist:
                            hf.create_dataset('type', shape=(n, 1), dtype=h5py.string_dtype())
                            hf['type'].attrs['num_processed'] = 0
                        
                        num_processed_images = hf[feature_name].attrs['num_processed']
                        if num_processed_images < n:
                            with h5py.File(hf_filename, 'a') as hf_preds:
                                for i in tqdm(range(num_processed_images, n), desc='Adding feature ' + str(feature_name)):
                                    hf[feature_name][i] = hf_preds['predicted'][i]
                                    hf[feature_name].attrs['num_processed'] = i+1
                            
                        if hf['type'].attrs['num_processed'] < n:
                            with h5py.File(hf_filename, 'a') as hf_preds:
                                for i in tqdm(range(hf['type'].attrs['num_processed'], n), desc='Adding types'):
                                    hf['type'][i] = hf_preds['type'][i]
                                    hf['type'].attrs['num_processed'] = i+1
                                types = hf['type'][:].reshape(-1)
                                for data_type in self.data_types:
                                    if data_type == 'all':
                                        hf['type'].attrs[data_type] = n
                                    else:
                                        hf['type'].attrs[data_type] = np.sum(data_type == types)

    def generate_model_comparisons(self, image_names=None, idx=None, dpi=None, regenerate=False):
        if image_names is None:
            image_names = [self.in_feature, self.out_feature] + self.get_all_model_types()
        
        with h5py.File(self.series_file, 'a') as hf:
            n = hf[self.in_feature].shape[0]
            if idx is None:
                idx = range(n)
            for img_idx in tqdm(idx, desc='Generating comparisons'):
                image_file_name = 'image_' + utils.pad_string(str(img_idx), length=4) + '.png'
                
                metric_save_folder_type = self.comparison_data_types_folders[hf['type'][img_idx][0]] / 'models'
                metric_save_folder_all = self.comparison_data_types_folders['all'] / 'models'
                metric_save_folder_type.mkdir(parents=True, exist_ok=True)
                metric_save_folder_all.mkdir(parents=True, exist_ok=True)
                
                file_name_type = metric_save_folder_type / image_file_name
                file_name_all = metric_save_folder_all / image_file_name
                if not regenerate and file_name_all.exists() and file_name_type.exists():
                    continue
                
                nrows, ncols = 2, len(image_names)
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(len(image_names)*2, 4))
                
                mask = 1 - hf['mask'][img_idx]
                
                images = {}
                for i in range(len(image_names)):
                    images[image_names[i]] = hf[image_names[i]][img_idx]
                i, j = 0, 0
                for row in ax:
                    for subplot in row:
                        subplot.axis('off')
                        if j == 0:
                            data = images[image_names[i]]
                            data_label_name = image_names[i]
                            color_map = 'viridis'
                            color_limits = (-2.5, 2.5)
                            color_bar_axes = [.87,.5325,.04,.35]
                        if j == 1:
                            data = images[image_names[i]] - images[self.out_feature]
                            data_label_name = image_names[i] + ' Diff'
                            color_map = 'seismic'
                            color_limits = (-2, 2)
                            color_bar_axes = [.87,.1225,.04,.35]
                            
                        if j == 1 and i == 0:
                            date_str = str(pd.to_datetime(hf['date'][img_idx],unit='s'))
                            subplot.text(0.2, 0, date_str, fontsize=12, ha='left', transform=subplot.transAxes)
                            
                        if not (j == 1 and i <= 1):
                            #data = hf[data_name][img_idx]
                            data = np.ma.masked_array(data, mask)
                            mesh = subplot.pcolormesh(data, cmap=color_map)
                            mesh.set_clim(color_limits[0], color_limits[1])
                            subplot.text(0.5, -0.1, data_label_name, fontsize=12, ha='center', transform=subplot.transAxes)
                            if i == ncols-1:
                                cax = fig.add_axes(color_bar_axes)
                                cax.axis('off')
                                fig.colorbar(mesh, ax=cax)
                        i += 1
                        
                    i, j = 0, j+1
                
                fig.savefig(file_name_type, dpi=dpi)
                fig.savefig(file_name_all, dpi=dpi)
                plt.close(fig)
         
    def calculate_metrics(self, metrics=['log_flux', 'flux', 'emission', 'MSE', 'SMSE', 'MAE', 'ME', 'PE', 'MSE-nolog', 'MAE-nolog'], remake=False):
        list_metrics = ['log_flux', 'flux']
        image_metrics = ['MSE', 'SMSE', 'MAE', 'ME', 'PE', 'emission', 'MSE-nolog', 'MAE-nolog']
        
        agg_types = ['mean', 'median']
        
        if self.temp_series_file.exists():
            os.remove(self.temp_series_file)
        
        with h5py.File(self.series_file, 'a') as hf:
            
            need_to_process = False
            image_names = [self.out_feature] + self.get_all_model_types()
            for data_type in self.data_types:
                n_type = hf['type'].attrs[data_type]
                for metric in metrics:
                    for img in image_names:
                        metric_name = data_type + '/' + metric + '/' + img
                        
                        #Remove previous datasets if it needs to remake
                        if remake:
                            if metric_name in hf:
                                del hf[metric_name]
                                
                        if metric in image_metrics:        
                            with h5py.File(self.temp_series_file, 'a') as hf_temp:
                                hf_temp.create_dataset(metric_name, shape=(n_type, self.img_size[0], self.img_size[1]))
                        
                        #Check to see if the dataset exists in the file
                        make_dataset = not metric_name in hf
                        
                        #Create the dataset if it doesn't exist
                        if make_dataset:
                                if metric in list_metrics:
                                    hf.create_dataset(metric_name, shape=(n_type,), chunks=True, dtype=np.float32)
                                if metric in image_metrics:
                                    hf.create_group(metric_name)
                                
                                hf[metric_name].attrs['num_processed'] = 0
                        
                        #See if additional processing needs to be done to process the dataset
                        computed_less = hf[metric_name].attrs['num_processed'] < n_type
                        need_to_process = need_to_process or computed_less
            
            if need_to_process:
                with h5py.File(self.temp_series_file, 'a') as hf_temp:
                    empty_model = self.load_empty_model()
                    inverse_normal, inverse_log = empty_model.transform_output_inverse
                    n_all = hf['type'].attrs['all']

                    type_indices = {}
                    for data_type in self.data_types:
                        type_indices[data_type] = 0

                    for i in tqdm(range(n_all), desc='Calculating metric values'):
                        mask = 1 - hf['mask'][i]
                        img_data_type = hf['type'][i][0]
                        true_img = hf[self.out_feature][i]

                        if any(m in metrics for m in ['PE', 'MSE-nolog', 'MAE-nolog']):
                            true_inv_img = inverse_log(inverse_normal(true_img))

                        for img in image_names:

                            #Save on multiple computations by pre-computing as necessary only once
                            if any(m in metrics for m in ['log_flux', 'flux', 'emission', 'PE', 'MSE-nolog', 'MAE-nolog']):
                                inv_normal_img = inverse_normal(hf[img][i])
                                inv_normal_img_mask = np.ma.masked_array(inv_normal_img, mask)
                            if any(m in metrics for m in ['flux', 'emission', 'PE', 'MSE-nolog', 'MAE-nolog']):
                                inv_log_img = inverse_log(inv_normal_img)
                                inv_log_img_mask = np.ma.masked_array(inv_log_img, mask)
                            if any(m in metrics for m in ['MSE', 'SMSE', 'MAE', 'ME']):
                                diff_data = hf[img][i] - hf[self.out_feature][i]
                            if any(m in metrics for m in ['PE', 'MSE-nolog', 'MAE-nolog']):
                                inverse_diff_data = true_inv_img - inv_log_img

                            for metric in metrics:
                                for data_type in self.data_types:
                                    n_type = hf['type'].attrs[data_type]
                                    metric_name = data_type + '/' + metric + '/' + img
                                    
                                    if data_type == img_data_type or data_type == 'all':
                                        if metric == 'log_flux':
                                            hf[metric_name][type_indices[data_type]] = inv_normal_img_mask.sum()
                                        elif metric == 'flux':
                                            hf[metric_name][type_indices[data_type]] = inv_log_img_mask.sum()
                                        elif metric == 'MSE':
                                            hf_temp[metric_name][type_indices[data_type]] = np.square(diff_data)
                                        elif metric == 'SMSE':
                                            hf_temp[metric_name][type_indices[data_type]] = np.multiply(np.sign(diff_data), np.square(diff_data))
                                        elif metric == 'MAE':
                                            hf_temp[metric_name][type_indices[data_type]] = np.abs(diff_data)
                                        elif metric == 'ME':
                                            hf_temp[metric_name][type_indices[data_type]] = diff_data
                                        elif metric == 'PE':
                                            hf_temp[metric_name][type_indices[data_type]] = 100 * np.divide(inverse_diff_data, true_inv_img)
                                        elif metric == 'MSE-nolog':
                                            hf_temp[metric_name][type_indices[data_type]] = np.square(inverse_diff_data)
                                        elif metric == 'MAE-nolog':
                                            hf_temp[metric_name][type_indices[data_type]] = np.abs(inverse_diff_data)
                                        elif metric == 'emission':
                                            hf_temp[metric_name][type_indices[data_type]] = inv_log_img_mask
                                        hf[metric_name].attrs['num_processed'] = type_indices[data_type] + 1


                        type_indices['all'] = type_indices['all'] + 1
                        type_indices[img_data_type] = type_indices[img_data_type] + 1

                    for data_type in self.data_types:
                        n_type = hf['type'].attrs[data_type]
                        for metric in metrics:
                            if metric in image_metrics:
                                for img in image_names:
                                    metric_name = data_type + '/' + metric + '/' + img
                                    raw_img_data = hf_temp[metric_name][:]
                                    
                                    for agg_type in ['mean', 'median']:
                                        agg_metric_name = metric_name + '/' + agg_type
                                        
                                        if agg_type == 'mean':
                                            img_data = np.mean(raw_img_data, axis=0)
                                            list_data = np.mean(raw_img_data, axis=(1, 2))
                                        elif agg_type == 'median':
                                            img_data = np.median(raw_img_data, axis=0)
                                            list_data = np.median(raw_img_data, axis=(1, 2))
                                        hf.create_dataset(agg_metric_name + '/img', shape=self.img_size)
                                        hf.create_dataset(agg_metric_name + '/list', shape=(n_type,), chunks=True, dtype=np.float32)
                                        hf[agg_metric_name + '/img'][:] = img_data
                                        hf[agg_metric_name + '/list'][:] = list_data
                                        hf[agg_metric_name].attrs['num_processed'] = n_type
                                        
            if self.temp_series_file.exists():
                os.remove(self.temp_series_file)
                            
        
    def find_flux_outliers(self, metric='flux', std_threshold=2.5, remake=False):
        outliers = {}
        for data_type in self.data_types:
            outliers[data_type] = []
            
        empty_model = self.load_empty_model()
        inverse_normal, inverse_log = empty_model.transform_output_inverse
        
        self.calculate_metrics(metrics=[metric], remake=remake)
        with h5py.File(self.series_file, 'a') as hf:
            n_all = hf['type'].attrs['all']
            for i in tqdm(range(n_all), desc='Looking for outliers'):
                mask = 1 - hf['mask'][i]
                
                img = None
                inv_normal_img = inverse_normal(hf[self.out_feature][i])
                if metric == 'log_flux':
                    img = inv_normal_img
                elif metric == 'flux':
                    img = inverse_log(inv_normal_img)
                img = np.ma.masked_array(img, mask)
                img_data_type = hf['type'][i][0]
                
                for data_type in self.data_types:
                    metric_name = data_type + '/' + metric + '/' + self.out_feature
                    flux_vals = hf[metric_name]
                    mean, std = np.average(flux_vals), np.std(flux_vals)
                    if data_type == img_data_type or data_type == 'all':
                        if np.abs((img.sum() - mean) / std) >= std_threshold:
                            outliers[data_type] = outliers[data_type] + [i]
        return outliers
        
    def generate_flux_images(self, metrics=['log_flux', 'flux'], remake=False, regenerate=False, dpi=None, std_scale_range=4):
        self.calculate_metrics(metrics=metrics, remake=remake)
        all_model_types = self.get_all_model_types()
        all_model_names = self.get_all_model_names()
        with h5py.File(self.series_file, 'r') as hf:
            with tqdm(total=len(self.data_types) * len(metrics), desc='Generating flux images') as pbar:
                for data_type in self.data_types:
                    for metric in metrics:
                        metric_save_folder = self.aggregate_data_types_folders[data_type] / metric
                        metric_save_folder.mkdir(parents=True, exist_ok=True)

                        comparison_file_name = metric_save_folder / 'comparison.png'
                        if regenerate or not comparison_file_name.exists():

                            n_model_variations = int(len(all_model_types) / len(self.model_types))
                            fig_tot, ax_tot = plt.subplots(nrows=n_model_variations, 
                                                           ncols=len(self.model_types), 
                                                           figsize=(4*len(self.model_types), 4*n_model_variations),
                                                           squeeze=False)

                            if metric == 'log_flux':
                                title = 'Total Flux of Log EUV Data'
                            if metric == 'flux':
                                title = 'Total Flux of EUV Data'

                            true_data = hf[data_type + '/' + metric + '/' + self.out_feature][:]
                            scale_mean, scale_std = np.average(true_data), np.std(true_data)
                            lowest_val = scale_mean - std_scale_range*scale_std
                            highest_val = scale_mean + std_scale_range*scale_std

                            val_range = (lowest_val, highest_val)
                            identity = np.linspace(val_range[0], val_range[1], 100)

                            for i in range(len(all_model_types)):
                                img = all_model_types[i]
                                r, c = i // len(self.model_types), i % len(self.model_types)

                                metric_name = data_type + '/' + metric + '/' + img
                                #true_data = np.array(df.at[metric_name, self.out_feature])
                                pred_data = hf[metric_name][:]
                                percent_error = np.average(np.abs(true_data - pred_data) / true_data)
                                
                                residual_func_intercept = lambda i: np.array(((true_data/1e8 - (pred_data/1e8 + i))**2).sum(), ndmin=1)
                                residual_func_intercept_prime = lambda i: np.array(-2*(true_data/1e8-(pred_data/1e8 + i)).sum(), ndmin=1)
                                residual_func_slope = lambda m: np.array(((true_data - (pred_data * m))**2).sum(), ndmin=1)
                                residual_func_slope_prime = lambda m: np.array((-2*pred_data*(true_data - (pred_data * m))).sum(), ndmin=1)
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    intercept = fsolve(residual_func_intercept, x0=[0], fprime=residual_func_intercept_prime)[0] * 1e8
                                    slope = fsolve(residual_func_slope, x0=[1], fprime=residual_func_slope_prime)[0]
                                percent_bias = percent_error - np.average(np.abs(true_data - (pred_data + intercept)) / true_data)
                                
                                systemic_error_percent = np.abs(slope - 1)
                                scatter_error_percent = np.average(np.abs(true_data - slope * pred_data) / true_data)
                                
                                
                                sub_title = title + ' with Model:\n' + all_model_names[i]
                                

                                fig, ax = plt.subplots(nrows=1, ncols=1)
                                ax.set_title(sub_title, y=1.02)
                                ax.set_xlabel('True')
                                ax.set_ylabel('Predicted')
                                ax.set_xlim(val_range[0], val_range[1])
                                ax.set_ylim(val_range[0], val_range[1])
                                ax.set_aspect('equal')
                                ax.plot(identity, identity, '--r')
                                ax.plot(identity, identity - intercept, ':b')
                                ax.scatter(x=true_data, y=pred_data, s=4**2, alpha=0.3, linewidths=0)
                                true_mean, true_std = np.average(true_data), np.std(true_data)
                                pred_mean, pred_std = np.average(pred_data), np.std(pred_data)

                                handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                                                 lw=0, alpha=0)] * 4
                                labels = []
                                labels.append("Percent Error : {0:.2f}".format(percent_error * 100))
                                #labels.append("True Mean/Std : {0:.2E}/{1:.2E}".format(true_mean, true_std))
                                #labels.append("Pred Mean/Std : {0:.2E}/{1:.2E}".format(pred_mean, pred_std))
                                labels.append("Percent Bias : {0:.2f}".format(percent_bias * 100))
                                labels.append("Percent Systemic Error : {0:.2f}".format(systemic_error_percent * 100))
                                labels.append("Percent Scatter Error : {0:.2f}".format(scatter_error_percent * 100))
                                ax.legend(handles, labels, loc='best', fontsize='medium', 
                                          fancybox=False, framealpha=0.7, 
                                          handlelength=0, handletextpad=0)

                                fig.savefig(metric_save_folder / self.get_aggregate_save_name(all_model_types[i], metric), dpi=dpi)
                                plt.close(fig)

                                ax_tot[r, c].set_xlabel('True')
                                ax_tot[r, c].set_ylabel('Predicted')
                                ax_tot[r, c].set_xlim(val_range[0], val_range[1])
                                ax_tot[r, c].set_ylim(val_range[0], val_range[1])
                                ax_tot[r, c].set_aspect('equal')
                                ax_tot[r, c].plot(identity, identity, '--r')
                                ax_tot[r, c].scatter(x=true_data, y=pred_data, s=4**2, alpha=0.3, linewidths=0)
                                ax_tot[r, c].text(0.5, 1.1, all_model_names[i], fontsize=12, ha='center', transform=ax_tot[r, c].transAxes)
                                ax_tot[r, c].legend(handles, labels, loc='best', fontsize='medium', 
                                          fancybox=False, framealpha=0.7, 
                                          handlelength=0, handletextpad=0)
                            fig_tot.tight_layout()
                            #plt.show()
                            fig_tot.savefig(metric_save_folder / 'comparison.png', dpi=dpi)
                            plt.close(fig_tot)
                        pbar.update(1)
                        
    def generate_flux_pair_images(self, metrics=['log_flux', 'flux'], model_type_pairs=['pixelwise', 'fcn8_dist_lat'], model_name_pairs=['Pixelwise', 'FCN with Limb and Lat'], remake=False, regenerate=False, dpi=None, std_scale_range=4, include_stats=False, show_trend_line=[False, True]):
        with h5py.File(self.series_file, 'r') as hf:
            for data_type in self.data_types:
                for metric in metrics:
                    metric_save_folder = self.aggregate_data_types_folders[data_type] / metric
                    metric_save_folder.mkdir(parents=True, exist_ok=True)

                    pair_file_name = metric_save_folder / ('pair_' + model_type_pairs[0] + '_' + model_type_pairs[1] + '.png')
                    if regenerate or not pair_file_name.exists():

                        fig_tot, ax_tot = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), squeeze=False)

                        true_data = hf[data_type + '/' + metric + '/' + self.out_feature][:]
                        scale_mean, scale_std = np.average(true_data), np.std(true_data)
                        lowest_val = scale_mean - std_scale_range*scale_std
                        highest_val = scale_mean + std_scale_range*scale_std

                        val_range = (lowest_val, highest_val)
                        identity = np.linspace(val_range[0], val_range[1], 100)

                        for i in range(2):
                            img = model_type_pairs[i]
                            r, c = 0, i

                            metric_name = data_type + '/' + metric + '/' + img
                            pred_data = hf[metric_name][:]
                            percent_error = np.average(np.abs(true_data - pred_data) / true_data)

                            residual_func_intercept = lambda i: np.array(((true_data/1e8 - (pred_data/1e8 + i))**2).sum(), ndmin=1)
                            residual_func_intercept_prime = lambda i: np.array(-2*(true_data/1e8-(pred_data/1e8 + i)).sum(), ndmin=1)
                            residual_func_slope = lambda m: np.array(((true_data - (pred_data * m))**2).sum(), ndmin=1)
                            residual_func_slope_prime = lambda m: np.array((-2*pred_data*(true_data - (pred_data * m))).sum(), ndmin=1)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                intercept = fsolve(residual_func_intercept, x0=[0], fprime=residual_func_intercept_prime)[0] * 1e8
                                slope = fsolve(residual_func_slope, x0=[1], fprime=residual_func_slope_prime)[0]
                            percent_bias = percent_error - np.average(np.abs(true_data - (pred_data + intercept)) / true_data)

                            #systemic_error_percent = np.abs(slope - 1)
                            scatter_error_percent = np.average(np.abs(true_data - slope * pred_data) / true_data)
                            
                            #ax.scatter(x=true_data, y=pred_data, s=4**2, alpha=0.3, linewidths=0)
                            true_mean, true_std = np.average(true_data), np.std(true_data)
                            pred_mean, pred_std = np.average(pred_data), np.std(pred_data)

                            ax_tot[r, c].set_xlabel('True')
                            ax_tot[r, c].set_ylabel('Predicted')
                            ax_tot[r, c].set_xlim(val_range[0], val_range[1])
                            ax_tot[r, c].set_ylim(val_range[0], val_range[1])
                            ax_tot[r, c].set_aspect('equal')
                            ax_tot[r, c].plot(identity, identity, '--r')
                            #ax_tot[r, c].plot(identity, identity - intercept, ':b')
                            ax_tot[r, c].scatter(x=true_data, y=pred_data, s=4**2, alpha=0.3, linewidths=0, c='k')
                            ax_tot[r, c].text(0.5, 1.05, model_name_pairs[i], fontsize=16, ha='center', va='bottom', transform=ax_tot[r, c].transAxes)
                            if include_stats:
                                handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                                                 lw=0, alpha=0)] * 4
                                labels = []
                                labels.append("Percent Error : {0:.2f}".format(percent_error*100))
                                labels.append("Percent Bias : {0:.2f}".format(percent_bias*100))
                                labels.append("Blue Slope : {0:.2f}".format(1/slope))
                                labels.append("Percent Scatter Error : {0:.2f}".format(scatter_error_percent * 100))
                                ax_tot[r, c].legend(handles, labels, loc='best', fontsize='medium', 
                                          fancybox=False, framealpha=0.7, 
                                          handlelength=0, handletextpad=0)
                            if show_trend_line[i]:
                                ax_tot[r, c].plot(identity * slope, identity, ':b')
                                
                        fig_tot.tight_layout()
                        #plt.show()
                        fig_tot.savefig(pair_file_name, dpi=dpi)
                        plt.close(fig_tot)     

    
    def generate_aggregate_comparison_images(self, metrics=['MSE', 'SMSE', 'MAE', 'ME', 'PE'], regenerate=False, dpi=None, std_colorbar_range=4, show_stats=False):
        all_model_types = self.get_all_model_types()
        all_model_names = self.get_all_model_names()
        agg_types = ['mean', 'median']
        
        with h5py.File(self.series_file, 'a') as hf:
            with tqdm(total=len(metrics)*len(self.data_types)*len(agg_types), desc='Generating aggregate comparisons') as pbar:
                for metric in metrics:
                    for data_type in self.data_types:
                        for agg_type in agg_types:
                        
                            highest_std = 0
                            for model_type in all_model_types:
                                metric_name = data_type + '/' + metric + '/' + model_type + '/' + agg_type + '/img'
                                highest_std = max(np.abs(np.std(hf[metric_name][:])), highest_std)

                            if metric in ['MSE', 'MAE', 'MSE-nolog', 'MAE-nolog']:
                                colormap = 'Reds'
                                color_limits = (0, std_colorbar_range * highest_std)
                            elif metric in ['SMSE', 'ME', 'PE']:
                                colormap = 'seismic'
                                color_limits = (-std_colorbar_range * highest_std, std_colorbar_range * highest_std)

                            metric_save_folder = self.aggregate_data_types_folders[data_type] / metric / agg_type
                            metric_save_folder.mkdir(parents=True, exist_ok=True)

                            comparison_file_name = metric_save_folder / 'comparison.png'

                            if regenerate or not comparison_file_name.exists():

                                n_model_variations = int(len(all_model_types) / len(self.model_types))
                                fig_tot, ax_tot = plt.subplots(nrows=n_model_variations, 
                                                               ncols=len(self.model_types),
                                                               squeeze=False, figsize=(2*(len(self.model_types)+1), 2*n_model_variations))
                                for i in range(len(all_model_types)):
                                    metric_name = data_type + '/' + metric + '/' + all_model_types[i] + '/' + agg_type + '/img'
                                    
                                    data = hf[metric_name][:]
                                    if metric == 'PE':
                                        mean_val = np.mean(np.abs(data))
                                        std_val = np.std(np.abs(data))
                                    else:
                                        mean_val = np.mean(data)
                                        std_val = np.std(data)
                                    mean_str = 'Mean: {0:.2E}\nSTD: {1:.2E}'.format(mean_val, std_val)
                                    mean_str_no_new_line = mean_str.replace('\n', ' ')

                                    r, c = i // len(self.model_types), i % len(self.model_types)

                                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
                                    ax.axis('off')
                                    ax.set_aspect('equal')
                                    mesh = ax.pcolormesh(data, cmap=colormap)
                                    mesh.set_clim(color_limits[0], color_limits[1])
                                    plt.colorbar(mesh,ax=ax)
                                    
                                    if show_stats:
                                        props = dict(boxstyle='square', facecolor='wheat', linewidth=0, alpha=0.15)
                                        ax.text(0.05, 0.95, mean_str, transform=ax.transAxes, fontsize=8, va='top', ha='left', bbox=props)
                                    fig.savefig(metric_save_folder / self.get_aggregate_save_name(all_model_types[i], metric), dpi=dpi)
                                    plt.close(fig)

                                    ax_tot[r, c].axis('off')
                                    ax_tot[r, c].set_aspect('equal')
                                    mesh_tot = ax_tot[r, c].pcolormesh(data, cmap=colormap)
                                    mesh_tot.set_clim(color_limits[0], color_limits[1])
                                    #ax_tot[r, c].text(0.5, 1.10, all_model_names[i], fontsize=8, ha='center', transform=ax_tot[r, c].transAxes)
                                    if show_stats:
                                        ax_tot[r, c].text(0.00, 1.00, mean_str_no_new_line, transform=ax_tot[r, c].transAxes, fontsize=6, va='bottom', ha='left', bbox=props)
                                
                                feature_names = ['None', 'Limb', 'Limb + Lat']
                                for r in range(3):
                                    ax_tot[r, 0].text(-0.05, 0.5, feature_names[r], transform=ax_tot[r,0].transAxes, fontsize=14, va='center', ha='right', rotation=90)
                                    
                                model_names = ['Pixelwise', 'U-Net', 'FCN']
                                for c in range(3):
                                    ax_tot[0, c].text(0.5, 1.05, model_names[c], transform=ax_tot[0,c].transAxes, fontsize=14, va='bottom', ha='center')
                                    
                                fig_tot.subplots_adjust(hspace=0.2, wspace=0.2)
                                fig_tot.colorbar(mesh_tot, ax=ax_tot.ravel().tolist())
                                fig_tot.savefig(metric_save_folder / 'comparison.png', dpi=dpi)
                                plt.close(fig_tot)
                            pbar.update(1)
                            
    def generate_feature_target_images(self, regenerate=False, idx=None, dpi=None, save_format='png', color_limit=1.75):
        names = ['He I', 'EUV']
        features = [self.in_feature, self.out_feature]

        with h5py.File(self.series_file, 'a') as hf:
            n = hf[self.in_feature].shape[0]
            if idx is None:
                idx = range(n)
            for img_idx in tqdm(idx, desc='Generating feature target images'):
                image_file_name = 'image_' + utils.pad_string(str(img_idx), length=4) + '.' + save_format
                
                metric_save_folder_type = self.comparison_data_types_folders[hf['type'][img_idx][0]] / 'feature_target'
                metric_save_folder_all = self.comparison_data_types_folders['all'] / 'feature_target'
                metric_save_folder_type.mkdir(parents=True, exist_ok=True)
                metric_save_folder_all.mkdir(parents=True, exist_ok=True)
                
                file_name_type = metric_save_folder_type / image_file_name
                file_name_all = metric_save_folder_all / image_file_name
                if not regenerate and file_name_all.exists() and file_name_type.exists():
                    continue
                
                
                mask = 1 - hf['mask'][img_idx]

                fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(6, 3))
                for i in range(2):
                    props = dict(boxstyle='round', facecolor=None, edgecolor=None, alpha=0)
                    ax[0, i].text(0.5, 0.0, names[i], transform=ax[0, i].transAxes, fontsize=12, va='top',ha='center', bbox=props)

                    ax[0, i].axis('off')
                    ax[0, i].set(adjustable='box', aspect='equal')
                    mesh = ax[0, i].pcolormesh(np.ma.masked_array(hf[features[i]][img_idx], mask), cmap='viridis')
                    mesh.set_clim(-color_limit, color_limit)
                fig.colorbar(mesh, ax=ax.ravel().tolist())
                fig.savefig(file_name_type, dpi=dpi)
                shutil.copy(str(file_name_type), str(file_name_all))
                plt.close(fig)
                
    def generate_feature_target_prediction_images(self, model_type, use_dist_feature=False, use_lat_feature=False, regenerate=False, idx=None, dpi=None, save_format='png', color_limit=1.75):
        names = ['He I', 'EUV', 'Prediction']
        model_name = self.get_model_save_name(model_type, use_dist_feature, use_lat_feature)
        features = [self.in_feature, self.out_feature, model_name]
        

        with h5py.File(self.series_file, 'a') as hf:
            n = hf[self.in_feature].shape[0]
            if idx is None:
                idx = range(n)
            for img_idx in tqdm(idx, desc='Generating feature target images'):
                image_file_name = 'image_' + utils.pad_string(str(img_idx), length=4) + '.' + save_format
                
                metric_save_folder_type = self.comparison_data_types_folders[hf['type'][img_idx][0]] / 'feature_target_prediction'
                metric_save_folder_all = self.comparison_data_types_folders['all'] / 'feature_target_prediction'
                metric_save_folder_type.mkdir(parents=True, exist_ok=True)
                metric_save_folder_all.mkdir(parents=True, exist_ok=True)
                
                file_name_type = metric_save_folder_type / image_file_name
                file_name_all = metric_save_folder_all / image_file_name
                if not regenerate and file_name_all.exists() and file_name_type.exists():
                    continue
                
                
                mask = 1 - hf['mask'][img_idx]

                fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(9, 3))
                for i in range(3):
                    props = dict(boxstyle='round', facecolor=None, edgecolor=None, alpha=0)
                    ax[0, i].text(0.5, 0.0, names[i], transform=ax[0, i].transAxes, fontsize=12, va='top',ha='center', bbox=props)

                    ax[0, i].axis('off')
                    ax[0, i].set(adjustable='box', aspect='equal')
                    mesh = ax[0, i].pcolormesh(np.ma.masked_array(hf[features[i]][img_idx], mask), cmap='viridis')
                    mesh.set_clim(-color_limit, color_limit)
                fig.colorbar(mesh, ax=ax.ravel().tolist())
                fig.savefig(file_name_type, dpi=dpi)
                shutil.copy(str(file_name_type), str(file_name_all))
                plt.close(fig)
                
    def generate_feature_target_prediction_difference_images(self, model_type, use_dist_feature=False, use_lat_feature=False, regenerate=False, idx=None, dpi=None, save_format='png', color_limit_hei=1.75, color_limit_euv=1.0, error_limit=0.75, top_left=(0, 0), crop_size=None, annotations=[]):
        names = ['He I', 'EUV', 'Error', 'Prediction']
        model_name = self.get_model_save_name(model_type, use_dist_feature, use_lat_feature)
        features = [self.in_feature, self.out_feature, model_name]

        with h5py.File(self.series_file, 'a') as hf:
            n = hf[self.in_feature].shape[0]
            if idx is None:
                idx = range(n)
            for img_idx in tqdm(idx, desc='Generating feature target images'):
                image_file_name = 'image_' + utils.pad_string(str(img_idx), length=4) + '.' + save_format
                
                metric_save_folder_type = self.comparison_data_types_folders[hf['type'][img_idx][0]] / 'feature_target_prediction_diff'
                metric_save_folder_all = self.comparison_data_types_folders['all'] / 'feature_target_prediction_diff'
                metric_save_folder_type.mkdir(parents=True, exist_ok=True)
                metric_save_folder_all.mkdir(parents=True, exist_ok=True)
                
                file_name_type = metric_save_folder_type / image_file_name
                file_name_all = metric_save_folder_all / image_file_name
                if not regenerate and file_name_all.exists() and file_name_type.exists():
                    continue
                
                
                mask = 1 - hf['mask'][img_idx]

                fig, ax = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(6, 6))
                i = 0
                text_boxes = []
                for r in range(2):
                    for c in range(2):
                        props = dict(boxstyle='round', facecolor=None, edgecolor=None, alpha=0)
                        name_idx = r*2+c
                        if r == 1 and c == 0:
                            data = np.ma.masked_array(hf[features[-1]][img_idx] - hf[self.out_feature][img_idx], mask)
                            if crop_size is not None:
                                data = data[top_left[0]:top_left[0]+crop_size[0],
                                            top_left[1]:top_left[1]+crop_size[1]]
                            mesh_err = ax[r, c].pcolormesh(data, cmap='seismic')
                            mesh_err.set_clim(-error_limit, error_limit)
                        else:
                            data = np.ma.masked_array(hf[features[i]][img_idx], mask)
                            if crop_size is not None:
                                data = data[top_left[0]:top_left[0]+crop_size[0],
                                            top_left[1]:top_left[1]+crop_size[1]]
                            mesh = ax[r, c].pcolormesh(data, cmap='viridis')
                            if c == 1:
                                mesh.set_clim(-color_limit_euv, color_limit_euv)
                            else:
                                mesh.set_clim(-color_limit_hei, color_limit_hei)
                            i += 1

                        text_box = ax[r, c].text(0.5, -0.03, names[name_idx], transform=ax[r, c].transAxes, fontsize=15, va='top',ha='center', bbox=props)
                        text_boxes.append(text_box)
                        ax[r, c].axis('off')
                        ax[r, c].set(adjustable='box', aspect='equal')
                        
                for annotation in annotations:
                    text_coord, arrow_coord, text = annotation
                    ax[0, 0].annotate(text,
                                      xy=arrow_coord, xycoords='axes fraction',
                                      xytext=text_coord, textcoords='axes fraction',
                                      arrowprops=dict(facecolor='white', shrink=0.05, edgecolor=None),
                                      bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                                      ha='right', va='center', color='black', fontsize=14)
                
                fig.tight_layout()
                cax_err = fig.add_axes([ax[0, 1].get_position().x1+0.2,ax[1, 1].get_position().y0,0.05,ax[0, 1].get_position().y1-ax[1, 1].get_position().y0])
                cax = fig.add_axes([ax[0, 1].get_position().x1+0.05,ax[1, 1].get_position().y0,0.05,ax[0, 1].get_position().y1-ax[1, 1].get_position().y0])

                #fig.colorbar(mesh_err, ax=ax.ravel().tolist())
                fig.colorbar(mesh_err, cax=cax_err)
                fig.colorbar(mesh, cax=cax)
                
                fig.savefig(file_name_type, dpi=dpi, bbox_extra_artists=text_boxes + [cax_err, cax], bbox_inches='tight')
                shutil.copy(str(file_name_type), str(file_name_all))
                plt.show()
                #plt.savefig(file_name_type, dpi=dpi)
                plt.close(fig)
    
    
    def generate_feature_images(self, regenerate=False, idx=None, dpi=None, save_format='png', color_limit=1.75):
        names = ['He I', 'Lim', 'Latitude']
        
        empty_model = empty_model = self.load_empty_model(add_distance_channel=True, add_latitude_channel=True)
        empty_model.dataset.set_transform(transform=empty_model.transform_eval)
        dataset = empty_model.dataset

        with h5py.File(self.series_file, 'a') as hf:
            n = hf[self.in_feature].shape[0]
            if idx is None:
                idx = range(n)
            for img_idx in tqdm(idx, desc='Generating feature target images'):
                image_file_name = 'image_' + utils.pad_string(str(img_idx), length=4) + '.' + save_format
                
                metric_save_folder_type = self.comparison_data_types_folders[hf['type'][img_idx][0]] / 'feature'
                metric_save_folder_all = self.comparison_data_types_folders['all'] / 'feature'
                metric_save_folder_type.mkdir(parents=True, exist_ok=True)
                metric_save_folder_all.mkdir(parents=True, exist_ok=True)
                
                file_name_type = metric_save_folder_type / image_file_name
                file_name_all = metric_save_folder_all / image_file_name
                if not regenerate and file_name_all.exists() and file_name_type.exists():
                    continue
                
                input_features = dataset[img_idx]['ew'].numpy()
                mask = 1 - hf['mask'][img_idx]

                fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(9, 3))
                for i in range(3):
                    props = dict(boxstyle='round', facecolor=None, edgecolor=None, alpha=0)
                    ax[0, i].text(0.5, 0.0, names[i], transform=ax[0, i].transAxes, fontsize=12, va='top',ha='center', bbox=props)

                    ax[0, i].axis('off')
                    ax[0, i].set(adjustable='box', aspect='equal')
                    mesh = ax[0, i].pcolormesh(np.ma.masked_array(input_features[i], mask), cmap='viridis')
                    mesh.set_clim(-color_limit, color_limit)
                fig.colorbar(mesh, ax=ax.ravel().tolist())
                fig.savefig(file_name_type, dpi=dpi)
                shutil.copy(str(file_name_type), str(file_name_all))
                plt.close(fig)
        
    def generate_all_output_images(self, make_series=True, remake=False, regenerate=False):
        if make_series:
            self.generate_baseline_series(remake=remake)
            self.generate_prediction_series()
        self.generate_flux_images(remake=remake, regenerate=regenerate)
        self.generate_aggregate_model_comparison_images(remake=remake, regenerate=regenerate)
        self.show_comparisons()
                    
    def calc_mse(self, model_type='unet', dist_channel=True):
        model_save_name = self.get_model_save_name(model_type, dist_channel) + '.pt'
        model = SunModel(device='gpu', save_folder=self.save_folder, save_name=model_save_name, load_model=True)

        val = np.min(model.training_history['validation loss'].values)
        return val
    
    
        
                        
            
            