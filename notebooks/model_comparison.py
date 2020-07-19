import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from torch.utils.data import DataLoader

from sunmodel import SunModel
from dataset import SunImageDataset
import utils

class ModelComparison(object):
    
    in_feature = 'ew'
    out_feature = '0304'
    
    data_types = ['all', 'train', 'valid', 'test']
    
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
            
    def settings_models(self, model_types=['unet', 'fcn8', 'fcn16', 'fcn32'], dropout=None):
        self.model_types = model_types
        self.dropout = dropout
        
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
        
        settings['epochs'] = self.epochs
        settings['batch_size'] = self.batch_size
        settings['num_steps_per_batch'] = self.num_steps_per_batch
        
        torch.save(settings, self.settings_file)
        
    def load_settings(self):
        settings = torch.load(self.settings_file)
        
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
                            dropout=settings['dropout'])
        self.settings_train(epochs=settings['epochs'],
                            batch_size=settings['batch_size'],
                            num_steps_per_batch=settings['num_steps_per_batch'])
        
    def get_model_save_name(self, model_type, dist_channel):
        save_name = model_type
        if dist_channel:
            save_name = save_name + '_dist'
        return save_name
        
    def get_model_prediction_save_name(self, model_type, dist_channel):
        save_name = self.get_model_save_name(model_type, dist_channel)
        return save_name + '_predictions.pkl'
    
    def get_entry_save_name(self, idx):
        return 'image_' + utils.pad_string(str(idx), pad='0', length=4) + '.pkl'
    
    def get_aggregate_save_name(self, model_type, metric):
        return model_type + '_' + metric + '.png'
    
    def get_all_model_types(self):
        all_model_types = []
        for dist_channel in [False, True]:
            for model_type in self.model_types:
                all_model_types.append(self.get_model_save_name(model_type, dist_channel))
        return all_model_types
        
    def load_model(self, model_type, dist_channel):
        model_save_name = self.get_model_save_name(model_type, dist_channel) + '.pt'
        model = SunModel(device='gpu', save_folder=self.save_folder, save_name=model_save_name, load_model=True)
        if not model.loaded_model:
            model.settings_model(model=model_type, add_distance_channel=dist_channel)
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
                                             random_state=self.random_state)
        return model
    
    def load_empty_model(self):
        model = SunModel(device='cpu', load_model=False)
        model.settings_model(add_distance_channel=False)
        model.settings_optimizer()
        model.settings_scheduler()
        model.settings_criterion()
        model.create_train_val_test_sets(val_proportion=self.val_proportion,
                                         test_proportion=self.test_proportion,
                                         random_state=self.random_state)
        return model
        
    
    def train_models(self):
        self.save_settings()
        for model_type in self.model_types:
            for dist_channel in [False, True]:
                
                print('Training model type ' + self.get_model_save_name(model_type, dist_channel))
                
                model = self.load_model(model_type, dist_channel)
                model.train_network(epochs=self.epochs)
                
                df = model.get_image_prediction(mode='all', include_original=False)
                df.to_pickle(self.save_folder / self.get_model_prediction_save_name(model_type, dist_channel))
                
                del model
                del df
                
    def generate_baseline_series(self, remake = False):
        empty_model = self.load_empty_model()
        empty_model.dataset.set_transform(transform=empty_model.transform_eval)
        data_loader = DataLoader(empty_model.dataset, batch_size=1, shuffle=False, pin_memory=True)
        
        idx = 0
        for batch in data_loader:
            '''
            filename = self.comparison_folder / self.get_entry_save_name(idx)
            if not filename.exists() or remake:
                in_img = np.array(batch[self.in_feature])
                out_img = np.array(batch[self.out_feature])
                
                mask = in_img[:, 0:1, :, :] != in_img[0, 0, 0, 0]
                out_img = out_img * mask

                new_entry = pd.Series({'idx': idx,
                                           self.in_feature: in_img.reshape(512, 512), 
                                           self.out_feature: out_img.reshape(512, 512)})

                print('Generating file ' + self.get_entry_save_name(idx))
                new_entry.to_pickle(self.comparison_folder / self.get_entry_save_name(idx))
            idx += 1
            '''
        del data_loader
        del empty_model
            
    def generate_prediction_series(self, remake_header_info=False):
        empty_model = self.load_empty_model()
        for model_type in self.model_types:
            for dist_channel in [True, False]:
                df_filename = self.save_folder / self.get_model_prediction_save_name(model_type, dist_channel)
                if df_filename.exists():
                    df = pd.read_pickle(df_filename)
                    for file in self.comparison_folder.iterdir():
                        if file.is_file():
                            series = pd.read_pickle(file)
                            d = df.loc[df['idx'] == series['idx']]

                            feature_name = self.get_model_save_name(model_type, dist_channel)
                            diff_feature_name = feature_name + '_diff'
                            changed_series = False

                            if any(name not in series.keys().to_list() for name in ['type', 'date']) or remake_header_info:
                                series['date'] = empty_model.dataset.get_date(series['idx'])
                                series['type'] = d['type'].values[0]
                                changed_series = True
                            if feature_name not in series.keys().to_list():
                                series[feature_name] = d['predicted'].values[0]
                                changed_series = True
                            if diff_feature_name not in series.keys().to_list():
                                #Pred - error = actual
                                series[diff_feature_name] = series[feature_name] - series['0304']
                                changed_series=True

                            if changed_series:
                                print('Added ' + feature_name + ' to file ' + str(file))
                                series.to_pickle(file)
                        
    def show_comparisons(self, image_names=None, idx=None, dpi=None, remake=False):
        if image_names is None:
            image_names = [self.in_feature, self.out_feature] + self.get_all_model_types()
        
        for file in self.comparison_folder.iterdir():
            #print(file)
            if file.is_file():
                file_name = str(file.parts[-1])
                file_id = int(file_name[6:-4])
                #print(file_id)
                if idx is not None and file_id not in idx:
                    continue
                series = pd.read_pickle(file)
                image_file_name = file.parts[-1][:-4] + '.png'
                
                file_name_type = self.comparison_data_types_folders[series['type']] / image_file_name
                file_name_all = self.comparison_data_types_folders['all'] / image_file_name
                if not remake and file_name_all.exists() and file_name_type.exists():
                    continue
                
                nrows, ncols = 2, len(image_names)
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(len(image_names)*2, 4))
                
                mask = series[self.in_feature] == series[self.in_feature][0, 0]
                
                i, j = 0, 0
                for row in ax:
                    for subplot in row:
                        subplot.axis('off')
                        if j == 0:
                            data_name = image_names[i]
                            data_label_name = image_names[i]
                            color_map = 'viridis'
                            color_limits = (-2.5, 2.5)
                            color_bar_axes = [.87,.5325,.04,.35]
                        if j == 1:
                            data_name = image_names[i] + '_diff'
                            data_label_name = image_names[i] + ' Diff'
                            color_map = 'seismic'
                            color_limits = (-2, 2)
                            color_bar_axes = [.87,.1225,.04,.35]
                            
                        if j == 1 and i == 0:
                            date_str = str(pd.to_datetime(series['date'],unit='s'))
                            subplot.text(0.2, 0, date_str, fontsize=12, ha='left', transform=subplot.transAxes)
                            
                        if not (j == 1 and i <= 1):
                            data = series[data_name]
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
                
                plt.show()
                
                fig.savefig(file_name_type, dpi=dpi)
                fig.savefig(file_name_all, dpi=dpi)
                plt.close(fig)
         
    def calculate_flux_values(self, remake=False):
        df_save_name = self.aggregate_folder / 'flux.pkl'
        if remake or not df_save_name.exists():
            print("Computing flux dataframe ...")
            image_names = [self.out_feature] + self.get_all_model_types()
                    
            metrics = ['log_flux', 'flux']
            all_metric_names = []
            for metric in metrics:
                for data_type in self.data_types:
                    metric_name = metric + '-' + data_type
                    all_metric_names.append(metric_name)
                    
            df = pd.DataFrame(index=all_metric_names, columns=image_names, dtype=object)
            df = df.fillna(0)
            df = df.astype(object)
            empty_model = self.load_empty_model()
            inverse_normal, inverse_log = empty_model.transform_output_inverse
            n = 0
            for file in self.comparison_folder.iterdir():
                if file.is_file():
                    n += 1
                    if n % 100 == 0:
                        print('Processed flux data for ' + str(n) + ' entries!')
                    series = pd.read_pickle(file)
                    mask = series[self.in_feature] == series[self.in_feature][0, 0]
                    for col in df.columns:
                        
                        inv_normal_img = inverse_normal(series[col])
                        inv_log_img = inverse_log(inv_normal_img)
                        inv_normal_img_mask = np.ma.masked_array(inv_normal_img, mask)
                        inv_log_img_mask = np.ma.masked_array(inv_log_img, mask)
                        
                        for metric in metrics:
                            for data_type in self.data_types:
                                metric_name = metric + '-' + data_type
                                if series['type'] == data_type or data_type == 'all':
                                    if df.at[metric_name, col] == 0:
                                        df.at[metric_name, col] = []
                                    if metric == 'log_flux':
                                        df.at[metric_name, col] = df.at[metric_name, col] + [inv_normal_img_mask.sum()]
                                    if metric == 'flux':
                                        df.at[metric_name, col] = df.at[metric_name, col] + [inv_log_img_mask.sum()]
            df.to_pickle(df_save_name)
            return df
        else:
            print("Loading precomputed flux dataframe ...")
            return pd.read_pickle(df_save_name)
        
    def find_flux_outliers(self, remake=False):
        metric = 'flux'
        std_threshold = 2.5
        
        outliers = {}
        for data_type in self.data_types:
            outliers[data_type] = []
            
        empty_model = self.load_empty_model()
        inverse_normal, inverse_log = empty_model.transform_output_inverse
        
        df = self.calculate_flux_values(remake=remake)
        n = 0
        for file in self.comparison_folder.iterdir():
                if file.is_file():
                    n += 1
                    if (n % 100) == 0:
                        print("Checked " + str(n) + " for outliers ...")
                    series = pd.read_pickle(file)
                    mask = series[self.in_feature] == series[self.in_feature][0, 0]
                    
                    inv_log_img = inverse_log(inverse_normal(series[self.out_feature]))
                    inv_log_img_mask = np.ma.masked_array(inv_log_img, mask)
                    
                    for data_type in self.data_types:
                        metric_name = metric + '-' + data_type
                        #print(series.index)
                        if series['type'] == data_type or data_type == 'all':
                            flux_vals = df.at[metric_name, self.out_feature]
                            mean, std = np.average(flux_vals), np.std(flux_vals)
                            if np.abs((inv_log_img_mask.sum() - mean) / std) >= std_threshold:
                                #file_name = str(file.parts[-1])
                                outliers[data_type] = outliers[data_type] + [series['idx']]
        return outliers
        
    def generate_flux_images(self, remake=False, regenerate=False):
        df = self.calculate_flux_values(remake=remake)
        all_model_types = self.get_all_model_types()
        for data_type in self.data_types:
            for metric in ['log_flux', 'flux']:
                metric_save_folder = self.aggregate_data_types_folders[data_type] / metric
                metric_save_folder.mkdir(parents=True, exist_ok=True)
                
                comparison_file_name = metric_save_folder / 'comparison.png'
                if regenerate or not comparison_file_name.exists():
                    
                    fig_tot, ax_tot = plt.subplots(nrows=2, ncols=len(all_model_types) // 2, figsize=(12, 9))

                    if metric == 'log_flux':
                        title = 'Total Flux of Log EUV Data'
                    if metric == 'flux':
                        title = 'Total Flux of EUV Data'
                    
                    lowest_val, highest_val = np.inf, 0
                    for i in range(len(all_model_types)):
                        metric_name = metric + '-' + data_type
                        true_data = np.array(df.at[metric_name, self.out_feature])
                        pred_data = np.array(df.at[metric_name, all_model_types[i]])
                        lowest_val = min(lowest_val, np.min(true_data), np.min(pred_data))
                        highest_val = max(highest_val, np.max(true_data), np.max(pred_data))
                    
                    val_range = (lowest_val*0.95, highest_val*1.05)
                    identity = np.linspace(val_range[0], val_range[1], 100)

                    for i in range(len(all_model_types)):
                        print("Generating comparison images for model: " + str(all_model_types[i]) 
                              + " with flux metric: " + str(metric) + " and data type: " + str(data_type) + " ...")
                        r, c = i // len(self.model_types), i % len(self.model_types)
                        metric_name = metric + '-' + data_type
                        true_data = np.array(df.at[metric_name, self.out_feature])
                        pred_data = np.array(df.at[metric_name, all_model_types[i]])
                        percent_error = np.abs(true_data - pred_data) / true_data

                        sub_title = title + ' with Model Type ' + all_model_types[i]

                        fig, ax = plt.subplots(nrows=1, ncols=1)
                        ax.set_title(sub_title)
                        ax.set_xlabel('True')
                        ax.set_ylabel('Predicted')
                        ax.set_xlim(val_range[0], val_range[1])
                        ax.set_ylim(val_range[0], val_range[1])
                        ax.set_aspect('equal')
                        ax.plot(identity, identity, '--r')
                        ax.scatter(x=true_data, y=pred_data, s=3**2)
                        true_mean, true_std = np.average(true_data), np.std(true_data)
                        pred_mean, pred_std = np.average(pred_data), np.std(pred_data)
                        
                        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                                         lw=0, alpha=0)] * 3
                        labels = []
                        labels.append("Percent Error : {0:.2f}".format(np.average(percent_error)*100))
                        labels.append("True Mean/Std : {0:.3E}/{1:.3E}".format(true_mean, true_std))
                        labels.append("Pred Mean/Std : {0:.3E}/{1:.3E}".format(pred_mean, pred_std))
                        ax.legend(handles, labels, loc='best', fontsize='medium', 
                                  fancybox=False, framealpha=0.7, 
                                  handlelength=0, handletextpad=0)
                        
                        print(true_data.shape)
                        
                        fig.savefig(metric_save_folder / self.get_aggregate_save_name(all_model_types[i], metric), dpi=300)
                        plt.close(fig)

                        ax_tot[r, c].set_xlabel('True')
                        ax_tot[r, c].set_ylabel('Predicted')
                        ax_tot[r, c].set_xlim(val_range[0], val_range[1])
                        ax_tot[r, c].set_ylim(val_range[0], val_range[1])
                        ax_tot[r, c].set_aspect('equal')
                        ax_tot[r, c].plot(identity, identity, '--r')
                        ax_tot[r, c].scatter(x=true_data, y=pred_data)
                        ax_tot[r, c].text(0.5, 1.1, all_model_types[i], fontsize=12, ha='center', transform=ax_tot[r, c].transAxes)
                        ax_tot[r, c].legend(handles, labels, loc='best', fontsize='medium', 
                                  fancybox=False, framealpha=0.7, 
                                  handlelength=0, handletextpad=0)
                    fig_tot.tight_layout()
                    fig_tot.savefig(metric_save_folder / 'comparison.png')
                    plt.close(fig_tot)
                else:
                    print("Skipping generation for flux metric: " + str(metric) + " and data type: " + str(data_type))
                        
                    
             
    def calculate_aggregate_model_comparison(self, metrics=['MSE', 'SMSE', 'MAE', 'ME'], remake=False):
        df_save_name = self.aggregate_folder / 'aggregate.pkl'
        if remake or not df_save_name.exists():
            print("Calculating aggregate dataframe ...")
            all_metrics = ['mean', 'std']
            for metric in metrics:
                for data_type in self.data_types:
                    all_metrics.append(metric + '-' + data_type)
            all_model_types = []
            for dist_channel in [False, True]:
                for model_type in self.model_types:
                    if dist_channel:
                        model_type = self.get_model_save_name(model_type, dist_channel)
                    all_model_types.append(model_type)

            df = pd.DataFrame(index=all_metrics, columns=[self.out_feature] + all_model_types, dtype=object)
            df = df.fillna(0)
            df = df.astype(object)
            n_train, n_val, n_test = 0, 0, 0
            for file in self.comparison_folder.iterdir():
                if file.is_file():
                    #print('Processing file ' + str(file) + '...')
                    series = pd.read_pickle(file)

                    if series['type'] == 'train':
                        n_train += 1
                    elif series['type'] == 'valid':
                        n_val += 1
                    elif series['type'] == 'test':
                        n_test += 1
                        
                    if (n_train + n_val + n_test) % 100 == 0:
                        print('Processed aggregate data for ' + str(n_train + n_val + n_test) + ' entries!')

                    for col in all_model_types:
                        for metric in metrics:
                            for data_type in self.data_types:
                                metric_name = metric + '-' + data_type
                                if series['type'] == data_type or data_type == 'all':
                                    #Mean Squared Error
                                    diff_data = series[col] - series[self.out_feature]
                                    if metric == 'MSE':
                                        df.at[metric_name, col] = df.at[metric_name, col] + np.square(diff_data)
                                    #Signed Mean Square Error
                                    elif metric == 'SMSE':
                                        df.at[metric_name, col] = df.at[metric_name, col] + np.multiply(np.sign(diff_data), np.square(diff_data))
                                    #Mean Absolute Error
                                    elif metric == 'MAE':
                                        df.at[metric_name, col] = df.at[metric_name, col] + np.abs(diff_data)
                                    #Mean Error
                                    elif metric == 'ME':
                                        df.at[metric_name, col] = df.at[metric_name, col] + diff_data

            for col in df.columns:
                print('Calculating mean and std for ' + col + '...')
                d_all = pd.DataFrame(columns=['type', col])
                for file in self.comparison_folder.iterdir():
                    if file.is_file():
                        series = pd.read_pickle(file)
                        new_series = pd.Series({'type': series['type'], col: series[col]})
                        d_all = d_all.append(new_series, ignore_index=True)
                for data_type in self.data_types:
                    if data_type == 'all':
                        d = d_all
                    else:
                        d = d_all.loc[d_all['type'] == data_type]
                    data = np.dstack(d[col].values)
                    data = np.rollaxis(data, -1)

                    df.at['mean-' + data_type, col] = np.mean(data)
                    df.at['std-' + data_type, col] = np.std(data)

                for metric in metrics:
                    for data_type in self.data_types:
                        metric_name = metric + '-' + data_type

                        if data_type == 'all':
                            df.at[metric_name, col] = df.at[metric_name, col] / (n_train + n_val + n_test)
                        elif data_type == 'train':
                            df.at[metric_name, col] = df.at[metric_name, col] / n_train
                        elif data_type == 'valid':
                            df.at[metric_name, col] = df.at[metric_name, col] / n_val
                        elif data_type == 'test':
                            df.at[metric_name, col] = df.at[metric_name, col] / n_test

            df.to_pickle(df_save_name)
            return df
        else:
            print("Loading precomputed aggregate dataframe ...")
            return pd.read_pickle(df_save_name)
           
    
    def generate_aggregate_model_comparison_images(self, metrics=['MSE', 'SMSE', 'MAE', 'ME'], remake=False, regenerate=False):
        df = self.calculate_aggregate_model_comparison(metrics, remake=remake)
        all_model_types = self.get_all_model_types()
        
        std_colorbar_range = 4
        
        for metric in metrics:
            
            highest_std = 0
            for data_type in self.data_types:
                for model_type in all_model_types:
                    data = df.at[metric + '-' + data_type, model_type]
                    highest_std = max(np.abs(np.std(data)), highest_std)
                    
            if metric in ['MSE', 'MAE']:
                colormap = 'Reds'
                color_limits = (0, std_colorbar_range * highest_std)
                
            elif metric in ['SMSE', 'ME']:
                colormap = 'seismic'
                color_limits = (-std_colorbar_range * highest_std, std_colorbar_range * highest_std)
                
            for data_type in self.data_types:
                metric_save_folder = self.aggregate_data_types_folders[data_type] / metric
                metric_save_folder.mkdir(parents=True, exist_ok=True)
                
                comparison_file_name = metric_save_folder / 'comparison.png'
                
                if regenerate or not comparison_file_name.exists():

                    fig_tot, ax_tot = plt.subplots(nrows=2, ncols=len(all_model_types) // 2)
                    for i in range(len(all_model_types)):
                        print("Generating comparison images for model: " + str(all_model_types[i]) + 
                              " with metric: " + str(metric) + " and data type: " + str(data_type) + " ...")
                        r, c = i // len(self.model_types), i % len(self.model_types)
                        data = df.at[metric + '-' + data_type, all_model_types[i]]

                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
                        ax.axis('off')
                        ax.set_aspect('equal')
                        mesh = ax.pcolormesh(data, cmap=colormap)
                        mesh.set_clim(color_limits[0], color_limits[1])
                        plt.colorbar(mesh,ax=ax)
                        fig.savefig(metric_save_folder / self.get_aggregate_save_name(all_model_types[i], metric), dpi=300)
                        plt.close(fig)

                        ax_tot[r, c].axis('off')
                        ax_tot[r, c].set_aspect('equal')
                        mesh_tot = ax_tot[r, c].pcolormesh(data, cmap=colormap)
                        mesh_tot.set_clim(color_limits[0], color_limits[1])
                        ax_tot[r, c].text(0.5, -0.1, all_model_types[i], fontsize=12, ha='center', transform=ax_tot[r, c].transAxes)
                        
                    fig_tot.colorbar(mesh_tot, ax=ax_tot.ravel().tolist())
                    fig_tot.savefig(metric_save_folder / 'comparison.png', dpi=300)
                    plt.close(fig_tot)
                else:
                    print("Skipping generation for metric: " + str(metric) + " and data type: " + str(data_type))
                
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
    
    
        
                        
            
            