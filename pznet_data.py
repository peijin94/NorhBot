"""
pznet_data.py
tools for the loading and preprocessing of the data
"""

__author__      = "Peijin Zhang"


from datetime import datetime, timedelta
import pytz
import sunpy
import sunpy.map
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.io import fits
import glob
from os import makedirs
import time
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import scipy.ndimage
from scipy.ndimage import zoom
import torch


from torch.utils.data import Dataset, DataLoader, random_split


# define a function to zoom
def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]
    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def cook_aia_fits(fname):
    """
    name : cook_aia_fits
    preprocessing function for the AIA data
    """
    datamap_sun = sunpy.map.Map(fname)
    new_dim = u.Quantity(datamap_sun.dimensions)/8
    desampled_map = datamap_sun.resample(new_dim)
    
    # float32 is enough and uses half memory and computational power as float64
    data_aia = desampled_map.data.astype(np.dtype('float32'))

    # move the solar centroid to the center of image
    ref_x = desampled_map.reference_coordinate.Tx.arcsec
    ref_y = desampled_map.reference_coordinate.Ty.arcsec
    ref_x_pix = desampled_map.reference_pixel.x.to_value()
    ref_y_pix = desampled_map.reference_pixel.y.to_value()
    r_sun_obs = desampled_map.rsun_obs.to_value()
    dx_asec = desampled_map.scale.axis1.to_value()
    dy_asec = desampled_map.scale.axis1.to_value()
    nax1 = desampled_map.dimensions.x.to_value()
    nax2 = desampled_map.dimensions.y.to_value()

    x = dx_asec*(np.arange(nax1)+1-ref_x_pix)+ ref_x
    y = dx_asec*(np.arange(nax1)+1-ref_x_pix)+ ref_y
    X,Y  = np.meshgrid(x,y)
    data_aia_centered = scipy.ndimage.shift(data_aia,(-ref_x/dx_asec,-ref_y/dy_asec))
    world_len = nax1*dx_asec
    scale_ratio = world_len/(r_sun_obs*2*1.1)

    # zoom the figure size to 1.1 R_s for the sun
    data_aia_zoom = scipy.ndimage.zoom(data_aia_centered,scale_ratio)
    padding_x = int((data_aia_zoom.shape[0]-nax1)/2)
    padding_y = int((data_aia_zoom.shape[1]-nax2)/2)

    data_aia_final = data_aia_zoom[ padding_x:(padding_x+512),padding_y:(padding_y+512)]
    
    return data_aia_final


def cook_norh17_fits(fname, nax1 = 512,nax2 = 512):
    """
    name : cook_norh17_fits
    preprocessing function for radioheliograph data (Norh 17GHz)
    """
    hdul = fits.open(fname)

    norh_header = hdul[0].header
    norh_data =   hdul[0].data

    x_c_idx  = norh_header['CRPIX1']
    y_c_idx  = norh_header['CRPIX2']

    x_idx_range = norh_header['NAXIS1']
    y_idx_range = norh_header['NAXIS2']

    dx = norh_header['CDELT1']
    dy = norh_header['CDELT2']

    x = (np.arange(x_idx_range)+1-x_c_idx)*dx
    y = (np.arange(y_idx_range)+1-y_c_idx)*dy

    X, Y = np.meshgrid(x, y)

    world_len =  norh_header['NAXIS1']*norh_header['CDELT1']
    scale_ratio = world_len/(norh_header['SOLR']*2*1.1)

    data_norh_zoom = scipy.ndimage.zoom(norh_data,scale_ratio)
    padding_x = int((data_norh_zoom.shape[0]-nax1)/2)
    padding_y = int((data_norh_zoom.shape[1]-nax2)/2)

    data_norh_final = data_norh_zoom[ padding_x:(padding_x+512),padding_y:(padding_y+512)]
    
    return data_norh_final


# construct the dataset !!!

class PZDataset(Dataset): 
    """
    name : PZDataset
    The dataset class for single input and single output
    """
    def __init__(self, dir_norh, dir_aia, 
                 wavelen = '304' ,  blacklist=[], transform=None):
        self.dir_norh = dir_norh
        self.dir_aia  = dir_aia
        self.wavelen = wavelen
        self.flist_norh = sorted(glob.glob(dir_norh+'*.npy'))
        self.fids = [x.split('/')[-1][-20:-4] for x in self.flist_norh]
        totallist = np.arange(len(self.flist_norh))
        masklist = totallist>-np.inf
        masklist[blacklist] = False
        self.finallist = totallist[masklist]
        
        self.transform = transform
        
    def __getitem__(self, index):
        cur_fid = self.fids[self.finallist[index]]
        img_aia  = np.load( 
            glob.glob(self.dir_aia+"*"+self.wavelen+"_"+cur_fid +"*")[0])
        img_norh = np.load( 
            glob.glob(self.dir_norh+"*"+cur_fid +"*")[0])
        
        img_aia[np.where(img_aia<1e-4)]=1e-4
        img_norh[np.where(img_norh<1e-4)]=1e-4
        
        img_aia[np.where(np.isnan(img_aia))]=1e-4
        img_norh[np.where(np.isnan(img_norh))]=1e-4
        
        img_aia = img_aia/1e3
        img_norh = img_norh/1e4
        
        if self.transform:
            img_aia = self.transform(img_aia)
            img_norh = self.transform(img_norh)
            
        return img_aia, img_norh, cur_fid+'_'+str(self.finallist[index])
    
    def __len__(self):
        return len(self.finallist)
    
    

class PZDatasetMC(Dataset): # multi_channel
    """
    name : PZDatasetMC
    The dataset class for multiple input and single output
    """
    def __init__(self, dir_norh, dir_aia, 
                 wavelen = ['304','211'] ,  blacklist=[], transform=None,
                 mem_all=False,size_img=(512,512),
                 aia_DB = 'aia_memDB_v1.npy',norh_DB = 'norh_memDB_v1.npy'):
        self.dir_norh = dir_norh
        self.dir_aia  = dir_aia
        self.wavelen = wavelen
        self.flist_norh = sorted(glob.glob(dir_norh+'*.npy'))
        self.fids = [x.split('/')[-1][-20:-4] for x in self.flist_norh]
        totallist = np.arange(len(self.flist_norh))
        masklist = totallist>-np.inf
        masklist[blacklist] = False
        self.finallist = totallist[masklist]
        self.num_ch_in = len(wavelen)
        self.transform = transform
        self.mem_all  = mem_all
        self.size_img = size_img
        
        if self.mem_all:
            self.NORH_memDB = np.load(norh_DB)
            self.AIA_memDB = np.load(aia_DB)
            
    def __getitem__(self, index):
        cur_fid = self.fids[self.finallist[index]]
        
        if not self.mem_all:
            img_norh_local = np.load(self.flist_norh[self.finallist[index]])
            img_aia  = torch.Tensor(len(self.wavelen),*img_norh_local.shape)
            img_norh = torch.Tensor(1,*img_norh_local.shape)
            img_norh[0,:,:] = torch.Tensor(img_norh_local)
            for ch_idx,ch_cur in enumerate(self.wavelen):
                try:
                    img_tmpload = np.load(self.dir_aia+"aia_"+ch_cur+"_"+cur_fid+'.npy')
                except:
                    print(self.finallist[index])
                img_tmp = torch.Tensor(img_tmpload)
                img_aia[ch_idx,:,:]  = img_tmp


            img_aia[np.where(img_aia<1e-6)]=1e-6
            img_norh[np.where(img_norh<1e-6)]=1e-6

            img_aia[np.where(np.isnan(img_aia))]=1e-6
            img_norh[np.where(np.isnan(img_norh))]=1e-6

            img_aia = img_aia/2e4
            img_norh = img_norh/1e4
        
        else:
            img_aia = torch.Tensor(self.AIA_memDB[index,:,:,:]).view(
                self.num_ch_in,*self.size_img)
            img_norh = torch.Tensor(self.NORH_memDB[index,:,:]).view(
                1,*self.size_img)
            
        
        
        if self.transform:
            img_aia = self.transform(img_aia)
            img_norh = self.transform(img_norh)
            
        return img_aia, img_norh, cur_fid+'_'+str(self.finallist[index])
    
    def __len__(self):
        return len(self.finallist)
    
    def prepare_tensorDB(self):
        """
        name : prepare_tensorDB
        when the size of dataset is not too large that it can fit in the memory of single node
        we recomend using this method to prepare a single big 4-D array contians all the training data
        The read-in time from disk can be saved
        """
        NORH_memDB = np.zeros([len(self.finallist),*self.size_img],dtype=np.float)
        AIA_memDB = np.zeros([len(self.finallist),self.num_ch_in,*self.size_img],dtype=np.float)
        
        for index in range(len(self.finallist)):
            cur_fid = self.fids[self.finallist[index]]
            img_norh = np.load(self.flist_norh[self.finallist[index]])
            img_aia  = np.zeros([len(self.wavelen),*img_norh.shape])
            
            for ch_idx,ch_cur in enumerate(self.wavelen):
                try:
                    img_tmpload = np.load(self.dir_aia+"aia_"+ch_cur+"_"+cur_fid+'.npy')
                except:
                    print(self.finallist[index])
                img_aia[ch_idx,:,:]  = img_tmpload
                
            img_aia[np.where(img_aia<1e-6)]=1e-6
            img_norh[np.where(img_norh<1e-6)]=1e-6

            img_aia[np.where(np.isnan(img_aia))]=1e-6
            img_norh[np.where(np.isnan(img_norh))]=1e-6

            img_aia = img_aia/2e4
            img_norh = img_norh/1e4
            
            NORH_memDB[index,:,:] = img_norh
            AIA_memDB[index,:,:,:] = img_aia
            
        np.save('norh_memDB_v1.npy',NORH_memDB)
        np.save('aia_memDB_v1.npy' ,AIA_memDB)
            

    
class AverageMeter(object):
    '''A handy class to measure the training status''' 
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
