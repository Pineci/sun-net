from astropy.io import fits
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import utils
from skimage.transform import resize

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Image(object):
    '''This class is a wrapper for image data. It isolates useful information for each image.
    Note: Not all of these fields are utilized, but these fields organized can help with
    future debugging or adding new features.
    
    Attributes
    ----------
    time_taken : pd.datetime
        The time that the image was taken
    data : np.array([[]])
        The np double array consisting of the image data, usually having dtype=float32
    image_type : str
        String representing the type of image. Examples include 'ew', '0304', '0171', 'cont', etc.
    file : pathlib.Path
        Path to the fits file containing the image data, can be used to extract more data if needed
        
    Methods
    ---------
    print_file_header_info()
        Prints information about the fits file header associated with the image
    
    '''
    
    def __init__(self, time_taken, data, image_type, file):
        '''Parameters
        ----------
        time_taken : pd.datetime
            The time the image was taken
        data : np.array([[]])
            The np double array consisting of the image data
        image_type : str
            String representing the type of image
        file : pathlib.Path
            Path to the file containing the image data
        '''
        self.time_taken = time_taken
        self.data = data
        self.image_type = image_type
        self.file = file
        
    def print_file_header_info(self):
        '''Prints the fits file header of the image associated with this
        image. Also fixes the file header to standard format if necessary.
        '''
        hdulist = fits.open(file)
        with fits.open(file) as hdulist: 
            hdulist.info()
            hdulist[1].verify('fix')
            #print(hdulist[0].header)
            #print(hdulist[1].header)

class SunImageDownloader(object):
    '''This class requires the HEI data to already be downloaded. Each downloaded image
    is processed by downloading the associated images of different wavelengths
    which are closest in time (up to 12 hours before or after the
    original HEII image was taken). Each entry is stored in a folder in a
    directory which contains all related fits files.
    
    Attributes
    ----------
    
    hei_path : pathlib.Path
        Path to the parent directory of the HEI data
    processed_path : pathlib.Path
        Path to the processed folder which will store every entry as a folder containing
        the associated fits files
    sdo_types : list(str)
        A list of strings representing the possible wavelengths of SDO images
        
    Example Usage
    -------------
    dl = SunImageDownloader()
    dl.copy_and_download_all_images(start_date='2010-05-13', end_date='2015-07-16')
    
    '''
    
    hei_path = pathlib.Path('../datasets/HEI_Images/solis.nso.edu/pubkeep/VSM 1083.0 (He I), level 2 (images) _v22')
    processed_path = pathlib.Path('../datasets/processed')
    
    sdo_types = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700', '4500']

    
    def get_hei_path(self, year=2014, month=5, day=7):
        '''Returns the path to the HEI folder for a given date. The default date
        is for testing purposes.
        
        Parameters
        ----------
        
        year : int, optional
            An integer representing the year which should be 4 digits (default = 2014)
        month : int, optional
            An integer representing the month from 1 to 12 inclusive (default = 5)
        day : int, optional
            An integer representing the day from 1 to 31 inclusive (default = 7)
            
        Returns
        -------
        
        path : pathlib.Path
            A path to the desired folder on the given date
        '''
        yr_str, month_str, day_str, _, _ = utils.get_date_strings(year=year, month=month, day=day)
        short_yr_str = yr_str[0] + yr_str[2:]
        return self.hei_path / (yr_str + month_str) / ('k4v2' + short_yr_str + month_str + day_str)
    
    
    def get_hei_files(self, year=2014, month=5, day=7):
        '''Returns a list of uncompressed or compressed HEI fits files on a given date.
        Only returns the list if the entry exists. The entry can be nonexistant if the
        date is invalid (e.g., February 31st) or no images were captured that day.
        The default date is for testing purposes. 
        
        Parameters
        ----------
        
        year : int, optional
            An integer representing the year which should be 4 digits (default = 2014)
        month : int, optional
            An integer representing the month from 1 to 12 inclusive (default = 5)
        day : int, optional
            An integer representing the day from 1 to 31 inclusive (default = 7)
            
        Returns
        -------
        
        files : list(pathlib.Path), None
            A list of paths to the HEI fits files on the specified date. If no images
            are found, returns None
        
        '''
        path = self.get_hei_path(year, month, day)
        if path.exists():
            return [x for x in path.iterdir() if x.suffix == '.gz' or x.suffix == '.fits']
    
    def copy_and_download_date_images(self, year=2014, month=5, day=7):
        '''Creates entries for a specific date in the self.processed_path folder of different 
        images of the sun. The entry is denoted by the folder name, consisting of a datetime 
        format which indicates the time that the HEI images were taken. The HEI files are 
        copied over from the HEI dataset folder, whereas the SDO images are downloaded from 
        the server on the fly. The images which are downloaded are the closest in time to 
        the capture time of the HEI images, up to a 12 hour window. If no images are found 
        within this time period, no images are downloaded. All wavelength images are downloaded, 
        even if not all may be used in the final model. The default date is for testing purposes.
    
        Parameters
        ----------
        
        year : int, optional
            An integer representing the year which should be 4 digits (default = 2014)
        month : int, optional
            An integer representing the month from 1 to 12 inclusive (default = 5)
        day : int, optional
            An integer representing the day from 1 to 31 inclusive (default = 7)
            
        '''
        #Get the associated fits files for the specified date
        files = self.get_hei_files(year=year, month=month, day=day)
        unique_times = []
        if files is not None:
            #Copy the HEI files
            for file in files:
                filename = file.parts[-1]
                yr_str, month_str, day_str, _, _ = utils.get_date_strings(year=year, month=month, day=day)

                #Get the time the image was taken
                parts = filename.split('_')
                time = parts[0].split('t')[1]
                hour = time[0:2]
                minute = time[2:4]
                second = time[4:6]
                image_date = pd.to_datetime(yr_str + '-' + month_str + '-' + day_str + '-' + hour + '-' + minute + '-' + second, 
                                            format='%Y-%m-%d-%H-%M-%S')
                
                #Stores the unique times that each image was downloaded, in the case
                #that multiple images were taken on the same day
                if image_date not in unique_times:
                    unique_times.append(image_date)
                
                #Copies the fits file to the desired folder (making the directory if necessary)
                new_folder = '../datasets/processed/' + image_date.strftime('%Y-%m-%d-H%H-M%M') + '/'
                copy_command = 'mkdir -p ' + str(new_folder) + ' && cp \"' + str(file) + '\" $_'
                os.system(copy_command)
                
            #Download the corresponding SDO files
            #Search for images within a 24 hours window of the time that the image was taken
            search_minute_deltas = [0]
            for t in range(1, 721):
                search_minute_deltas.append(t)
                search_minute_deltas.append(-1*t)
            for wavelength_str in self.sdo_types:
                for time in unique_times:
                    for time_delta in search_minute_deltas:

                        new_time = time + pd.Timedelta(time_delta, unit='m')
                        yr_str, month_str, day_str, hour_str, minute_str = utils.get_date_strings(year=new_time.year, 
                                                                                                  month=new_time.month, 
                                                                                                  day=new_time.day, 
                                                                                                  hour=new_time.hour, 
                                                                                                  minute=new_time.minute)

                        filename = 'AIA' + yr_str + month_str + day_str + '_' + hour_str + minute_str + '_' + wavelength_str + '.fits'
                        image_search_path = 'http://jsoc2.stanford.edu/data/aia/synoptic/' + yr_str + '/' + month_str + '/' + day_str
                        image_search_path += '/' + 'H' + hour_str + '00/' + filename
                        
                        #Entry format is based on the time it was taken
                        save_path = '../datasets/processed/' + time.strftime('%Y-%m-%d-H%H-M%M') + '/'
                        new_file_path = pathlib.Path(save_path + filename)
                        
                        #Download the desired file if it doesn't already exist, halt the search if
                        #the image was downloaded since then we found the closest temporal image
                        if new_file_path.exists():
                            break
                        os.system('wget -nv ' + image_search_path + ' -P ' + save_path)
                        if new_file_path.exists():
                            break
                            
    def copy_and_download_all_images(self, start_date="2000-01-01", end_date="2020-12-31"):
        '''Copy and download all images within the specified date range, inclusive. Uses the method
        copy_and_download_date_images. It is recommended to tighten the search range to as short of
        a time period as possible to prevent excessive searching for image files in which the image
        capturing was not active.
        
        Parameters
        ----------
        start_date : str, optional
            The first possible date to start processing entries (default = '2000-01-01')
        end_date : str, optional
            The last possible date to process entries (default = '2020-12-31')
        
        '''
        start_date_pd = pd.to_datetime(start_date)
        end_date_pd = pd.to_datetime(end_date)
        current_day = start_date_pd
        while current_day <= end_date_pd:
            self.copy_and_download_date_images(year=current_day.year, month=current_day.month, day=current_day.day)
            current_day = current_day + pd.Timedelta(1, unit='d')
    
class SunImageDataset(Dataset):
    '''This class implments torch's dataset class to facilitate use with dataloaders.
    This class reads the files stored in the processed folder, and reads only
    the '0304' and 'ew' image types (the former is the output image, the latter is
    the input image). The resulting images are stored in two tensors.
    
    Attributes
    ----------
    
    processed_path : pathlib.Path
        The folder path to the data entries
    hei_types : list(str)
        Possible image types for HEI images
    sdo_types : list(str)
        Possible image types for SDO images, represented by wavelengths
    transform : dict(torchvision.transforms), None
        Torchvision transforms, which are applied to the tensors before returning.
        The transforms are stored in a dictionary, where each dictionary entry stores
        the transformation to be applied to the image type which is stored as the key
        If None, no transformation is applied
    x_tensor : torch.tensor
        Torch tensor which stores the input images as a [N, 1, W, H] tensor, where N, W, H
        represent the number of images, width, and height respectively
    y_tensor : torch.tensor
        Torch tensor which stores the output images as a [N, 1, W, H] tensor, where N, W, H
        are described in x_tensor
    x_type : str
        The image type to be stored in the x_tensor
    y_type : str
        The image type to be stored in the y_tensor
    
    '''
    
    processed_path = pathlib.Path('../datasets/processed')
    
    hei_types = ['cont', 'ew', 'color']
    sdo_types = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700', '4500']
    
    x_type = 'ew'
    y_type = '0304'
    
    def __init__(self, transform=None, remake=False):
        '''
        Parameters
        ----------
        
        transform: torchvision.transforms, None
            Transformations to be applied to the tensors
        remake: bool
            If true, re-reads all images in the processed directory. Otherwise,
            loads the tensors from previously computed files to speed up processing
            time.
        '''
        self.transform = transform
        self.x_tensor = None
        self.y_tensor = None
        self.get_all_images(remake=remake)
        
    def set_transform(self, transform=None):
        '''Sets the transform of the class.
        
        Parameters
        ----------
        
        transform : dict(torchvision.transforms), None
            Transform to be applied to the tensors before returning. Each dictionary entry
            must indicate an image type to which the transformation will be applied. If None,
            no transformation is applied
        '''
        self.transform = transform
                  
                
    def get_images(self, folder, allowed_types):
        '''Loads the images for a given folder in the processed directory. Only
        reads the image if the Image's image_type is in allowed_types to prevent
        unnecessary file reads.
        
        Parameters
        ----------
        
        folder : pathlib.Path
            A path to the folder which contains fits files for a certain time. Should
            have the same name format as created by the SunImageDownloader class
        allowed_types : list(str)
            A list of available image_types to read
            
        Returns
        -------
        
        images : list(Images)
            A list of the images read from the folder, each element has type Image (defined in this
            file)
        
        '''
        images = []
        for file in [x for x in folder.iterdir() if x.suffix == '.gz' or x.suffix == '.fits']:
            filename = file.parts[-1]
            
            #Get the image type
            image_type = None
            if file.suffix == '.gz':
                filename = filename[0:-7]
            else:
                filename = filename[0:-5]
            types = filename.split('_')
            if len(types) > 1:
                image_type = types[-1]
            else:
                image_type = 'color'
                
            #Only proceed to process the image if it is of an allowed type
            if image_type in allowed_types:
                #Get the time the image was taken
                image_date = None
                if image_type in self.hei_types:
                    image_date = pd.to_datetime(folder.parts[-1], format="%Y-%m-%d-H%H-M%M")
                elif image_type in self.sdo_types:
                    image_date = pd.to_datetime(filename, format="AIA%Y%m%d_%H%M_" + image_type)

                #Get the image data
                #First, get the card which contains the image data
                image_card_id = None
                if image_type in self.hei_types:
                    image_card_id = 0
                elif image_type in self.sdo_types:
                    image_card_id = 1

                #Read the data from the card
                image_data = None
                with fits.open(file) as hdulist:
                    hdulist[image_card_id].verify('fix')
                    image_data = hdulist[image_card_id].data
                if len(image_data.shape) == 3:
                    image_data = np.transpose(image_data, (1, 2, 0))    
                new_image = Image(image_date, image_data, image_type, file) 
                images.append(new_image)
        return images
    
    
    def read_all_images(self, allowed_types=['ew', '0304'], new_size=[512, 512]):
        '''This function reads all entries in the processed folder. It stores
        the resulting images in a pandas dataframe which contains a column for
        each image type, along with the date of that entry, which can be thought
        of as a unique identifier for the data point. This function also removes
        a number of pixels from the '0304' image type so that the sun takes up the
        same amount of space in the image frame as the 'ew' image type. All images
        are rescaled to a uniform size. If something goes wrong in this process,
        i.e. something went wrong with the reading process, the program skips the
        entry and moves on.
        
        Parameters
        ----------
        
        allowed_types : list(str)
            A list of strings indicating the allowed image types
        new_size : [int, int]
            A list representing the rescaled [width, height] of the processed images
            
        Returns
        -------
        
        df : pd.DataFrame
            A dataframe containg columns of 'allowed_types' and a date column
        '''
        df = pd.DataFrame()
        i = 1
        for folder in self.processed_path.iterdir():
            try:
                entry_date = pd.to_datetime(folder.parts[-1], format="%Y-%m-%d-H%H-M%M")
                image_dict = {}
                image_dict['date'] = entry_date
                for image in self.get_images(folder, allowed_types):
                    data = image.data
                    #Remove 80 pixels to have the sun take about the same space in the frame
                    #as the 'ew' images
                    if image.image_type == '0304':
                        height, width = data.shape
                        pixel_remove_width = 80
                        data = data[pixel_remove_width:(height-pixel_remove_width), pixel_remove_width:(width-pixel_remove_width)]
                    image_dict[image.image_type] = resize(data, (new_size[0], new_size[1]))
                    image_dict[image.image_type] = image_dict[image.image_type].astype(dtype='float32')
                image_series = pd.Series(image_dict)
                df = df.append(image_series, ignore_index=True)
                print('Dataframe shape: ' + str(df.shape) + '\tFolder: ' + str(i))
            except:
                print('Failed to read folder: ' + str(i))
            i += 1
        return df
    
    def get_all_images(self, remake=False):
        '''Loads all the image data as tensors. To speed up reading time, a copy of the
        tensor is saved to disk to facilitate loading the images again. If remake
        is True, then all images are read again rather than reading the tensor. This
        function also gets rid of any entries where at least one of the input or output
        image does not exist or contains NaNs. A new axis is also added indicating that these
        images contain only 1 channel.
        
        Parameters
        ----------
        remake : boolean, optional
            If false and the saved tensor files already exist, then the tensor files are loaded.
            Otherwise, the images are all re-read
        
        '''
        save_location = pathlib.Path('../datasets')
        if save_location.exists() and not remake:
            self.x_tensor = torch.load(save_location / 'x_images.pt')
            self.y_tensor = torch.load(save_location / 'y_images.pt')
        else:
            df = self.read_all_images()
            df = df.dropna()
            self.x_tensor = torch.tensor(np.array(list(df[self.x_type].values), dtype=np.float32), dtype=torch.float32)
            self.y_tensor = torch.tensor(np.array(list(df[self.y_type].values), dtype=np.float32), dtype=torch.float32)
            torch.save(self.x_tensor, save_location / 'x_images.pt')
            torch.save(self.y_tensor, save_location / 'y_images.pt')
            
        to_keep = []
        for i in range(len(self)):
            arr_x = np.array(self.x_tensor[i]).reshape((512, 512))
            arr_y = np.array(self.y_tensor[i]).reshape((512, 512))
            if not np.isnan(arr_x.mean()) and not np.isnan(arr_y.mean()):
                to_keep.append(i)
        
        self.x_tensor = self.x_tensor[to_keep, :, :]
        self.y_tensor = self.y_tensor[to_keep, :, :]
        self.x_tensor = self.x_tensor[:, None, :, :]
        self.y_tensor = self.y_tensor[:, None, :, :]
    
    def __len__(self):
        '''Returns the number of entries in the dataset, i.e., the number of data
        points.
        
        Returns
        -------
        
        len : int
            Number of data points
        '''
        return list(self.x_tensor.shape)[0]
    
    def __getitem__(self, idx):
        '''Returns a slice of the tensors according to the input indices along the data point axis. 
        Also applies the classes transform to the data before returning. The tensors are returned
        as a dictionary, where keys denote the image type.
        
        Paramters
        ---------
        
        idx : int, list(int), tensor(int)
            The indices of the sliced tensor. Each entry in idx should be an integer 
            between 0 and len(self)
        
        Returns
        -------
        
        sample : dictionary(torch.tensor)
            A dictionary where keys denote image type, and the values are the desired
            tensor slices
        
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = {self.x_type: self.x_tensor[idx], self.y_type: self.y_tensor[idx]}
        
        if self.transform is not None:
            for key in sample.keys():
                sample[key] = self.transform[key](sample[key])
                
        return sample