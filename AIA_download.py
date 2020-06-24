
"""
AIA_download.py
download the aia data according to the Norh datetime.
"""

__author__      = "Peijin Zhang"

from datetime import datetime, timedelta
import pytz
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.io import fits
import glob
from os import makedirs
import time

wavelen_all = [1700,193,304,131,171,211,94,335,1600,193] # all the wavelengths
wavelen_select = [211,171,335,193]  # selected wavelengths
wavelen_range_select = [9,9,9,9]
sdo_dir = 'data_sdo/'
verbose=False

# get file list of Norh
flist = sorted(glob.glob('./data_norh/*'))


# read in Norh data, download the corresponding AIA data.
t = time.time()
for findex in range(len(flist)):
    #(1840,len(flist)):
    print('downloading file ['+str(findex)+']'+' of '+str(len(flist)))
    fname = flist[findex][-16:]
    print(fname)
    # get datetime from fits and convert to UTC
    fdata = fits.open(flist[findex])
    t_str = fdata[0].header['jstdate']+' '+fdata[0].header['jsttime']
    japan_t = datetime.strptime(t_str,'%Y-%m-%d %H:%M:%S.%f')
    japan_timezone = pytz.timezone('Asia/Tokyo')
    jp_time = japan_timezone.localize(japan_t,is_dst=None)
    ut_time = jp_time.astimezone(pytz.utc)
    corr_dir = sdo_dir+fname+'/' # a dir for each Norh file
    makedirs(corr_dir, exist_ok=True)
    idx_wl = 0

    for wavelength in wavelen_select:
        attrs_time = a.Time(ut_time-timedelta(0,wavelen_range_select[idx_wl])
                    , ut_time+timedelta(0,wavelen_range_select[idx_wl]))
        result = Fido.search(attrs_time, a.Instrument('aia'),
                             a.Wavelength(wavelength*u.angstrom))
        wavelength_dir = corr_dir+str(wavelength)+'/'
        n_t_expand = 1
        while (result.file_num<0.5) and (n_t_expand<6): # try five times before giving up
            n_t_expand = n_t_expand+1
            attrs_time = a.Time(ut_time-n_t_expand*timedelta(0,wavelen_range_select[idx_wl])
                        , ut_time+n_t_expand*timedelta(0,wavelen_range_select[idx_wl]))
            result = Fido.search(attrs_time, a.Instrument('aia'),
                                 a.Wavelength(wavelength*u.angstrom))
            wavelength_dir = corr_dir+str(wavelength)+'/'


        makedirs(wavelength_dir, exist_ok=True)
        dfiles = Fido.fetch(result,path = wavelength_dir, #[0,int(result.file_num/2)]
                           progress=verbose, overwrite=True)
        print(result)

        idx_wl = idx_wl+1

    tmp_elapsed = time.time() - t
    print('time Elapsed : '+str(int(tmp_elapsed))+'s')

elapsed = time.time() - t
print(elapsed)
