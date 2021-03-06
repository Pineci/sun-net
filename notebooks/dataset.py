from astropy.io import fits
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import utils
import h5py
import shutil
import sunpy.map
from tqdm import tqdm
from skimage.transform import resize
from collections import Sequence

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
    transform : torchvision.transforms, None
        Torchvision transforms, which are applied to the tensors before returning.
        The transforms should operate on a dictionary, where each dictionary entry stores
        stores theh image type, a custom transform class would be helpful here
        If None, no transformation is applied
    x_data : np.array
        Numpy array which stores the input images as a [N, 1, W, H] tensor, where N, W, H
        represent the number of images, width, and height respectively
    y_data : np.array
        Numpy array which stores the output images as a [N, 1, W, H] tensor, where N, W, H
        are described in x_data
    x_type : str
        The image type to be stored in the x_data
    y_type : str
        The image type to be stored in the y_data
    
    '''
    
    
    hei_types = ['cont', 'ew', 'color']
    sdo_types = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700', '4500']
    
    x_type = 'in'
    y_type = 'out'
    
    excluded_dates = ['2013-04-17-H18-M24',
                      '2012-08-23-H21-M55',
                      '2012-10-12-H19-M22',
                      '2012-12-18-H19-M40',
                      '2014-04-13-H17-M02',
                      '2015-07-08-H18-M38',
                      '2011-11-02-H18-M14',
                      '2012-01-18-H18-M53',
                      '2012-09-21-H22-M01',
                      '2010-09-01-H16-M51',
                      '2011-10-25-H17-M57',
                      '2011-05-07-H18-M48',
                      '2011-12-24-H18-M24',
                      '2011-10-16-H17-M08',
                      '2011-10-18-H17-M10',
                      '2012-11-18-H21-M37',
                      '2011-11-14-H20-M21',
                      '2012-08-23-H22-M08',
                      '2012-07-16-H21-M00',
                      '2012-10-12-H20-M01',
                      '2010-08-21-H22-M09',
                      '2011-03-15-H17-M28',
                      '2011-09-25-H18-M38',
                      '2011-01-27-H19-M51',
                      '2013-01-02-H19-M27',
                      '2013-03-26-H22-M25',
                      '2012-11-16-H20-M36',
                      '2011-03-18-H18-M22',
                      '2011-03-17-H19-M20',
                      '2011-10-12-H21-M09',
                      '2010-11-02-H17-M47',
                      '2012-01-11-H20-M45',
                      '2011-10-23-H19-M35',
                      '2011-04-19-H16-M19',
                      '2013-03-28-H16-M37',
                      '2011-03-14-H16-M46',
                      '2011-05-12-H16-M55',
                      '2013-04-22-H16-M05',
                      '2011-03-03-H18-M09',
                      '2011-09-15-H18-M14',
                      '2011-03-25-H19-M36',
                      '2011-07-18-H18-M14',
                      '2011-10-09-H17-M57',
                      '2011-01-13-H18-M10',
                      '2012-07-10-H17-M29',
                      '2011-03-05-H15-M41',
                      '2011-10-07-H18-M26',
                      '2010-09-07-H18-M22',
                      '2011-03-12-H16-M55',
                      '2011-11-14-H20-M47',
                      '2012-10-18-H20-M27',
                      '2012-08-23-H21-M30',
                      '2010-09-07-H17-M07',
                      '2012-09-21-H19-M48',
                      '2012-03-26-H17-M58',
                      '2013-01-23-H20-M56',
                      '2011-01-11-H17-M30',
                      '2011-04-07-H17-M21',
                      '2010-10-12-H17-M27',
                      '2012-10-11-H16-M17',
                      '2010-12-01-H17-M28',
                      '2011-05-19-H17-M19',
                      '2010-11-04-H18-M05',
                      '2011-09-13-H20-M10',
                      '2010-11-03-H19-M35',
                      '2013-04-10-H19-M29',
                      '2012-08-23-H21-M43',
                      '2010-06-30-H18-M40',
                      '2010-11-28-H18-M04',
                      '2012-10-17-H17-M11',
                      '2013-11-28-H18-M05',
                      '2011-05-19-H17-M07',
                      '2012-11-02-H19-M00',
                      '2010-10-11-H16-M48',
                      '2011-06-26-H18-M06',
                      '2011-10-13-H17-M31',
                      '2010-10-02-H18-M16',
                      '2011-10-18-H17-M54',
                      '2010-09-07-H15-M53']
    
    #Values from https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/response/aia_V8_20171210_050627_response_table.txt
    interpolation_values = {'2010-03-24T00:00:00.000': 0.10424,
                           '2011-01-27T15:00:00.000': 0.06483,
                           '2011-04-13T18:00:00.000': 0.05197,
                           '2011-05-20T18:00:00.000': 0.04532,
                           '2011-10-06T12:00:00.000': 0.03270,
                           '2012-01-01T12:00:00.000': 0.02977,
                           '2012-04-10T12:00:00.000': 0.03533,
                           '2013-02-15T12:00:00.000': 0.03471,
                           '2013-05-01T12:00:00.000': 0.03394,
                           '2013-10-01T12:00:00.000': 0.02817,
                           '2014-05-25T12:00:00.000': 0.02059,
                           '2015-05-01T12:00:00.000': 0.01128,
                           '2016-05-01T12:00:00.000': 0.00702}
    
    
    def __init__(self, transform=None, remake=False, correct_sensor_data=True, load_into_memory=False, in_types=['ew'], out_types=['0304'], folder_indices=None, remake_images=False, data_source='hard-disk'):
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
        self.in_types = in_types
        self.out_types = out_types
        self.processed_path = pathlib.Path('../datasets/processed')
        self.save_folder = pathlib.Path('../datasets')
        self.save_name = 'images.h5'
        self.save_location = self.save_folder / self.save_name
        self.load_into_memory = load_into_memory
        self.data_source = data_source
        if data_source == 'hard-disk':
            self.access_location = self.save_location
        elif data_source == 'ssd':
            root_path = pathlib.Path('/mnt/scratch/nfs_fs02/apineci')
            root_path.mkdir(parents=False, exist_ok=True)
            self.access_location = root_path / self.save_name
            
        self.get_all_images(remake=remake, 
                            correct_sensor_data=correct_sensor_data, 
                            in_types=self.in_types, 
                            out_types=self.out_types,
                            folder_indices=folder_indices,
                            remake_images=remake_images)
        
            
        
    def set_transform(self, transform=None):
        '''Sets the transform of the class.
        
        Parameters
        ----------
        
        transform : torchvision.transforms, None
            Transform to be applied to the tensors before returning. If None, no 
            transformation is applied
        '''
        self.transform = transform
        
    def get_interpolation_value(self, date_timestamp):
        start_dates = list(map(lambda x: pd.to_datetime(x[:-4], format='%Y-%m-%dT%H:%M:%S').timestamp(), self.interpolation_values.keys()))
        eff_area = list(self.interpolation_values.values())
        return np.interp(date_timestamp, start_dates, eff_area)
            
    def check_valid_fits_file(self, file, img_type):
        try:
            if img_type in self.sdo_types:
                with fits.open(file) as hdulist:
                    hdulist[1].verify('fix')
                    quality = hdulist[1].header['QUALITY']
                    exptime = hdulist[1].header['EXPTIME']
                    dataskew = hdulist[1].header['DATASKEW']
                    datakurt = hdulist[1].header['DATAKURT']
                    valid = quality == 0
                    if img_type == '0304':
                        valid = valid and np.abs(exptime-3) <= 1
                    return valid
            else:
                return True
        except:
            return False
        
    def get_image_type(self, file):
        filename = file.parts[-1]
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
        return image_type, filename
        
    def check_valid_fits_folder(self, folder, allowed_types):
        valid = True
        found_types = []
        for file in [x for x in folder.iterdir() if x.suffix == '.gz' or x.suffix == '.fits']:
            image_type, _ = self.get_image_type(file)
            found_types.append(image_type)
            if image_type in allowed_types:
                valid = valid and self.check_valid_fits_file(file, image_type)
        return valid and all([t in found_types for t in allowed_types])
                  
                
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
            image_type, filename = self.get_image_type(file)
            
            #m = sunpy.map.Map(file)
            #data = m.data
            #print(data.shape)
            #for item in m.meta:
            #    print("Key: {} Val: {}".format(item, m.meta[item]))
            #print(m.meta)
            #break
                
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
                    #hdulist.info()
                    hdulist[image_card_id].verify('fix')
                    #print(hdulist[image_card_id].header)
                    image_data = hdulist[image_card_id].data
                if len(image_data.shape) == 3:
                    image_data = np.transpose(image_data, (1, 2, 0))   
                #print(image_type)
                #print(hdulist[image_card_id].header.keys)
                
                if image_type in self.hei_types:
                    exposure_time = hdulist[image_card_id].header['STOPTIME'] - hdulist[image_card_id].header['STARTIME']
                elif image_type in self.sdo_types:
                    exposure_time = hdulist[image_card_id].header['EXPTIME']
                    
                image_data = image_data / exposure_time
                new_image = Image(image_date, image_data, image_type, file) 
                images.append(new_image)
        return images
    
    
    def read_all_images(self, save_folder, save_name, in_types=['ew'], out_types=['0304'], new_size=(864, 864), remake=False, folder_indices=None, correct_sensor_data=True, remake_images=False):
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
        save_location = save_folder / save_name
        temp_location = save_folder / 'temp.h5'

        allowed_types = in_types + out_types
        remake = remake or not save_location.exists()
        if remake:
            if save_location.exists():
                save_location.unlink()
            if temp_location.exists():
                temp_location.unlink()
            i, created_dataset = -1, False
            num_folders = sum([1 for _ in self.processed_path.iterdir()])
            with tqdm(total=num_folders, desc='Processing dataset folders') as pbar:
                with h5py.File(temp_location, 'w') as hf:
                    data_shape = (num_folders, new_size[0], new_size[1])
                    hf.create_dataset('date', shape=(num_folders, ))
                    for type_name in allowed_types:
                        hf.create_dataset(type_name, shape=data_shape, dtype=np.float32)
                        #hf[type_name].attrs['date'] = image_dict['date'].value
                with h5py.File(temp_location, 'a') as hf:
                    for folder in self.processed_path.iterdir():
                        pbar.update(1)
                        try:
                            i += 1
                            if folder_indices is not None:
                                if i < folder_indices[0] or i > folder_indices[1]:
                                    continue
                            if not self.check_valid_fits_folder(folder, allowed_types):
                                continue
                            entry_date = pd.to_datetime(folder.parts[-1], format="%Y-%m-%d-H%H-M%M")
                            image_dict = {}
                            image_dict['date'] = entry_date
                            hf['date'][i] = image_dict['date'].value / 10 ** 9
                            for image in self.get_images(folder, allowed_types):
                                data = image.data
                                #Remove 80 pixels to have the sun take about the same space in the frame
                                #as the 'ew' images
                                if image.image_type in ['0171', '0304']:
                                    height, width = data.shape
                                    pixel_remove_width = 80
                                    data = data[pixel_remove_width:(height-pixel_remove_width), pixel_remove_width:(width-pixel_remove_width)]
                                if correct_sensor_data and image.image_type == '0304':
                                    data = data * 1/self.get_interpolation_value(hf['date'][i])
                                image_dict[image.image_type] = resize(data, (new_size[0], new_size[1]))
                                image_dict[image.image_type] = image_dict[image.image_type].astype(dtype='float32')
                            #print("Read Folder Images!")
                            n_read = 0

                            
                            for type_name in allowed_types:
                                hf[type_name][i, :, :] = image_dict[type_name]
                                n_read = hf[type_name].shape[0]
                            #print("Wrote images to hf!")
                            #hf['date'].resize((hf['date'].shape[0] + 1), axis=0)
                            #hf['date'][-1:] = image_dict['date'].value / 10 ** 9
                            #for type_name in allowed_types:
                            #    hf[type_name].resize((hf[type_name].shape[0] + 1), axis=0)
                            #    print(hf[type_name][-1:].shape)
                            #    print(image_dict[type_name].shape)
                            #    hf[type_name][-1:] = image_dict[type_name]
                            #    n_read = hf[type_name].shape[0]
                        
                                        #print('Type Name: ' + type_name + ' Shape: ' + str(hf[type_name].shape))
                            #print("Read " + str(n_read) + " images!")

                        except KeyboardInterrupt:
                            print("Quitting...")
                            break
                    
                #except:
                #    print('Failed to read folder: ' + str(i))
                
            #print("Read " + str(n_read) + " images!")
        if remake_images or remake:
            if save_location.exists():
                save_location.unlink()
            excluded_dates_list = list(map(lambda x: pd.to_datetime(x, format="%Y-%m-%d-H%H-M%M").value / 10 ** 9, self.excluded_dates))

            to_keep = []
            eps = 1e-7
            with h5py.File(temp_location, 'a') as hf:
                n_read = hf['date'].shape[0]
                for idx in tqdm(range(n_read), desc='Checking for bad images'):
                    date = hf['date'][idx]
                    matching_times = list(filter(lambda x: np.abs(x) < 100, list(map(lambda x: x - date, excluded_dates_list))))
                    date_check = len(matching_times) == 0 and date != 0
                    #print(date_check)
                    nan_check, std_check = True, True
                    for type_name in allowed_types:
                        arr = hf[type_name][idx]
                        nan_check = nan_check and not np.isnan(arr.mean())
                        std_check = std_check and np.std(arr) > eps
                    if nan_check and std_check and date_check:
                        to_keep.append(idx)
                        
                
                with h5py.File(save_location, 'w') as hf_save:
                    data_shape = (len(to_keep), 1, new_size[0], new_size[1])
                    hf_save.create_dataset('date', shape=(len(to_keep),), chunks=True)
                    for in_type in in_types:
                            hf_save.create_dataset(in_type, shape=data_shape, dtype=np.float32, chunks=True)
                    for out_type in out_types:
                            hf_save.create_dataset(out_type, shape=data_shape, dtype=np.float32, chunks=True)
                    for i in tqdm(range(len(to_keep)), desc='Copying good images'):
                        for in_type in in_types:
                            hf_save[in_type][i, 0, :, :] = hf[in_type][to_keep[i], :, :]
                        for out_type in out_types:
                            hf_save[out_type][i, 0, :, :] = hf[out_type][to_keep[i], :, :]
                        hf_save['date'][i] = hf['date'][to_keep[i]]
                    '''
                    in_data_shape = (len(to_keep), len(in_types), new_size[0], new_size[1])
                    out_data_shape = (len(to_keep), len(out_types), new_size[0], new_size[1])
                    hf_save.create_dataset('x', shape=in_data_shape, dtype=np.float32, chunks=True)
                    hf_save.create_dataset('y', shape=out_data_shape, dtype=np.float32, chunks=True)
                    hf_save.create_dataset('date', shape=(len(to_keep),), chunks=True)
                    for i in range(len(to_keep)):
                        for in_type_idx in range(len(in_types)):
                            hf_save['x'][i, in_type_idx, :, :] = hf[in_types[in_type_idx]][to_keep[i], :, :]
                        for out_type_idx in range(len(out_types)):
                            hf_save['y'][i, out_type_idx, :, :] = hf[out_types[out_type_idx]][to_keep[i], :, :]
                        hf_save['date'][i] = hf['date'][to_keep[i]]
                    '''
                    print("Kept " + str(len(to_keep)) + " images!")
    
    def get_all_images(self, remake=False, correct_sensor_data=True, size=(864, 864), in_types=['ew'], out_types=['0304'], folder_indices=None, remake_images=False):
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
        #self.dates_data, self.x_data, self.y_data = self.read_all_images(self.save_folder, self.save_name, new_size=size, remake=remake)
        self.read_all_images(self.save_folder, self.save_name, new_size=size, remake=remake, in_types=in_types, out_types=out_types, folder_indices=folder_indices, remake_images=remake_images)
        if self.load_into_memory:
            with h5py.File(self.save_location, 'a') as hf:
                self.x_data = {}
                self.y_data = {}
                for in_type in self.in_types:
                    self.x_data[in_type] = hf[in_type][:]
                for out_type in self.out_types:
                    self.y_data[out_type] = hf[out_type][:]
        if self.data_source == 'ssd':
            if not self.access_location.exists():
                shutil.copy(self.save_location, self.access_location)
        with h5py.File(self.access_location, 'a') as hf:
            self.dates_data = hf['date'][:]
            self.length = hf[self.in_types[0]].shape[0]
        
                    
    
    def __len__(self):
        '''Returns the number of entries in the dataset, i.e., the number of data
        points.
        
        Returns
        -------
        
        len : int
            Number of data points
        '''
        return self.length
    
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
        #print(idx)
        #print(type(idx))
        #print(isinstance(idx, Sequence))
        concat_axis = 1 if isinstance(idx, Sequence) or isinstance(idx, slice)  or isinstance(idx, np.ndarray) else 0
        

        sample_x, sample_y = [], []
        if self.load_into_memory:
            for in_type in self.in_types:
                sample_x.append(self.x_data[in_type][idx])
            for out_type in self.out_types:
                sample_y.append(self.y_data[out_type][idx])
        else:
            with h5py.File(self.access_location, 'a') as hf:
                for in_type in self.in_types:
                    sample_x.append(hf[in_type][idx])
                for out_type in self.out_types:
                    sample_y.append(hf[out_type][idx])
        
        sample = {}
        sample[self.x_type] = None
        sample[self.y_type] = None
        for i in range(len(sample_x)):
            data = sample_x[i]
            #if concat_axis == 1:
            #    data = data[None, :, :, :]
            if sample[self.x_type] is None:
                sample[self.x_type] = data
            else:
                sample[self.x_type] = np.concatenate((sample[self.x_type], data), axis=concat_axis)
        for i in range(len(sample_y)):
            data = sample_y[i]
            #if concat_axis == 1:
            #    data = data[None, :, :, :]
            if sample[self.y_type] is None:
                sample[self.y_type] = data
            else:
                sample[self.y_type] = np.concatenate((sample[self.y_type], data), axis=concat_axis)
        
        if self.transform is not None:
            sample = self.transform(sample)
                
        return sample
    
    def get_date(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return self.dates_data[idx]
    
    def show_images_and_names(self):
        
        save_location = pathlib.Path('../datasets')
        filename = 'df_data.pkl'
        file_path = save_location / filename
        if not file_path.exists():
            df = self.read_all_images()
            df = df.dropna()
            df.to_pickle(file_path)
        else:
            df = pd.read_pickle(file_path)
            
        columns = 2
        rows = 1
        
        for index, row in df.iterrows():
            print('Image: ' + str(index) + ' Date: ' + str(row['date']))
            in_img = row['ew']
            out_img = row['0304']
        
            fig=plt.figure(figsize=(8, 8))
            fig.add_subplot(rows, columns, 1)
            plt.imshow(in_img.reshape(512, 512))
            fig.add_subplot(rows, columns, 2)
            plt.imshow(out_img.reshape(512, 512))
            plt.show()