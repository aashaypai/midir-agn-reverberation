
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries, aggregate_downsample
import astropy.units as u
import os
#from lightcurve_processing import nsigma_clipping, local_nsigma_clipping

project_root = os.path.abspath(os.path.join(os.path.dirname('mid-ir-agn-dust-reverberation'), ".."))
IMPORT_FILEPATH = os.path.join(project_root, "data")


manga_wise_file = fits.open(IMPORT_FILEPATH  + "/manga-wise-variable-0.3.0.fits")
manga_wise_hdu = manga_wise_file[1]
mwv = manga_wise_hdu.data

##########################################################################################################
def nsigma_clipping(ts, n):
    """perform N-sigma clipping on timeseries to remove points that lie outside N standard deviations from the median

    Parameters
    ----------

    ts : astropy.Timeseries Object

    Returns
    -------

    ts : astropy.Timeseries Object

    """
    mask = ((ts['mag'] < np.nanmedian(ts['mag'])+n*np.std(ts['mag'])) & (ts['mag'] > np.nanmedian(ts['mag'])-n*np.std(ts['mag'])))

    ts = ts[mask]

    return ts



def local_nsigma_clipping(ts, n, size=5):
    """perform n-sigma clipping on points using the median of local values to remove points that lie outside n standard deviations from the median

    Parameters
    ----------

    ts : astropy.Timeseries Object

    n : np.float64
        number of standard deviations to clip outside of
    
    size : np.int32
        number of points to consider to calculate local median 

    Returns
    -------

    ts_new : astropy.Timeseries Object

    """
    clipped_count = 0

    time = ts['time'].to_value('decimalyear')
    mag = ts['mag']
    mag_err = ts['mag_err']

    time_new, mag_new, mag_err_new = np.array([]), np.array([]), np.array([])

    time_new = np.append(time_new, time[:size])
    mag_new = np.append(mag_new, mag[:size])
    mag_err_new = np.append(mag_err_new, mag_err[:size])
    
    for i, obs in enumerate(mag):
        
        if i < size or i > mag.shape[0] - size:
            continue

        
        med = np.nanmedian(mag[i-size:i+size])
        
        std = np.std(mag[i-size:i+size])
        
        #print('median:', str(med))
        
        if ((obs < med + n*std) and (obs > med - n*std)):
            time_new = np.append(time_new, time[i])
            mag_new = np.append(mag_new, mag[i])
            mag_err_new = np.append(mag_err_new, mag_err[i])
        
        else:
            clipped_count +=1 

    time_new = np.append(time_new, time[-1*size:])
    mag_new = np.append(mag_new, mag[-1*size:])
    mag_err_new = np.append(mag_err_new, mag_err[-1*size:])

    #print('points removed:', str(clipped_count))

    tb_new = Table()
    tb_new['mag'] = mag_new
    tb_new['mag_err'] = mag_err_new

    ts_new = TimeSeries(tb_new, time=Time(time_new, format='decimalyear'))
    return ts_new
def import_manga(i, j, k): 
    """Import MaNGA data: MNSA, MaNGA-WISE-variables, Pipe3D

    Parameters
    ----------

    i : np.int
        HDU number for MNSA file

    j : np.int
        HDU number for MaNGA-WISE-variable file

    k : np.int
        HDU number for Pipe3D file

    Returns
    -------

    mnsa_hdu : FITS HDU
    
    manga_wise_hdu : FITS HDU

    pipe3d_hdu : FITS HDU

    """
    manga_file = fits.open(IMPORT_FILEPATH + "/mnsa-0.3.2.fits")
    mnsa_hdu = manga_file[i]
    mnsa = mnsa_hdu.data

    manga_wise_file = fits.open(IMPORT_FILEPATH + "/manga-wise-variable-0.3.0.fits")
    manga_wise_hdu = manga_wise_file[j]
    mwv = manga_wise_hdu.data

    pipe3d_file = fits.open(IMPORT_FILEPATH +  '/SDSS17Pipe3D_v3_1_1.fits')
    pipe3d_hdu = pipe3d_file[k]
    pipe3d = pipe3d_hdu.data

    return mnsa_hdu, manga_wise_hdu, pipe3d_hdu


def import_lightcurves(crts_name, asassn_name, ztf_name, ptf_name):
    """Import lightcurve data: CRTS, ASAS-SN, ZTF, PTF

    Parameters
    ----------
    crts_name1 : string
        file name of crts 1 file

    crts_name2 : string
        file name of crts 2 file

    asassn_name : string
        file name of ASAS-SN file

    ztf_name : string
        file name of ZTF file

    ptf_name : string
        file name of PTF file

    Returns
    -------

    crts : astropy.Timeseries Object

    asassn : astropy.Timeseries Object

    ztf : astropy.Timeseries Object

    ptf : astropy.Timeseries Object

    """
    crts = TimeSeries.read(IMPORT_FILEPATH + '/Lightcurves/CRTS/' + crts_name, format='csv', time_column='MJD', time_format='mjd')
    #crts2 = TimeSeries.read(IMPORT_FILEPATH + '/Lightcurves/' + crts_name2, format='csv', time_column='MJD', time_format='mjd')
    #crts = astropy.table.vstack([crts1, crts2])

    crts = crts[crts['InputID']=='8553-1901']
    asassn = TimeSeries.read(IMPORT_FILEPATH + '/Lightcurves/ASAS-SN/' + asassn_name, format='ascii', time_column='JD', time_format='jd')

    ztf = TimeSeries.read(IMPORT_FILEPATH + '/Lightcurves/ZTF/' + ztf_name, format='csv', time_column='mjd', time_format='mjd')
    
    ptf = TimeSeries.read(IMPORT_FILEPATH + '/Lightcurves/PTF/' + ptf_name, format='csv', time_column='obsmjd', time_format='mjd')
    
    return crts, asassn, ztf, ptf


def process_wise(plateifu, lc, band=2):
    """process WISE lightcurve data

    Parameters
    ----------

    plateifu : string
        MaNGA plate-IFU identifier of the galaxy

    lc : astropy.fits file
        FITS file containing lightcurve data
    Returns
    -------

    ts : astropy.Timeseries Object

    """
    lc = lc[lc['plateifu'] == plateifu]

    tb = Table()

    if band == 2:
        tb['mag'] = lc['mean_W2_per_epoch'][0]
        tb['mag_err'] = lc['err_W2_per_epoch'][0]
    
    elif band == 1:
        tb['mag'] = lc['mean_W1_per_epoch'][0]
        tb['mag_err'] = lc['err_W1_per_epoch'][0]
        
    ts = TimeSeries(tb, time=Time(lc['mjd'][0], format='mjd'))
    
    mask = (np.isfinite(ts['mag']))

    ts = ts[mask]


    return ts

def process_crts(ts, clip=None):
    """process CRTS lightcurve data

    Parameters
    ----------

    ts : astropy.Timeseries Object

    Returns
    -------

    ts : astropy.Timeseries Object

    """
    ts.sort(['InputID', 'time'])

    plateifuv = mwv[mwv['var_flag']==1]['plateifu']
    inds = np.array([])
    for i, pifu in enumerate(plateifuv):
        obj = ts[ts['InputID']==pifu]
        index = i*np.ones(obj['InputID'].shape[0])
        inds = np.append(inds, index)
    ts['Index'] = inds

    #unsure??
    ts = ts[ts['Blend']==0]
    
    ts.keep_columns(['time', 'Index', 'Mag', 'Magerr'])

    ts.rename_column('Mag', 'mag')
    ts.rename_column('Magerr', 'mag_err')

    if clip != None:
        if ts['mag'].shape[0] > 0:
            ts = nsigma_clipping(ts, clip)
            ts = local_nsigma_clipping(ts, clip)
    return ts



def process_asassn(ts, filter, clip=None):
    """process ASAS-SN lightcurve data

    Parameters
    ----------

    ts : astropy.Timeseries Object

    filter : string
        'V' for V band, 'g' for g band

    Returns
    -------

    ts : astropy.Timeseries Object

    """
    ts.sort(['time'])

    ts = ts[ts['Filter']==filter]

    mask = ((ts['Mag Error'].value > 99))
    ts = ts[~mask]

    ts['Mag'] = ts['Mag'].astype(np.float64)

    ts.rename_column('Mag', 'mag')
    ts.rename_column('Mag Error', 'mag_err')

    ts.keep_columns(['time', 'mag', 'mag_err'])

    if clip != None:
        if ts['mag'].shape[0] > 0:
            ts = nsigma_clipping(ts, clip)
            ts = local_nsigma_clipping(ts, clip)

    return ts

def process_ztf(ts, filter, clip=None):
    """process ASAS-SN lightcurve data

    Parameters
    ----------

    ts : astropy.Timeseries Object

    filter : string
        'zg' for g band, 'zr' for R band, 'zi' for i band
        
    Returns
    -------

    ts : astropy.Timeseries Object

    """
    ts.sort(['time'])

    ts = ts[ts['filtercode']==filter]
    ts = ts[ts['catflags'] == 0]
    ts.keep_columns(['time', 'mag', 'magerr'])

    ts.rename_column('magerr', 'mag_err')

    if clip != None:
        if ts['mag'].shape[0] > 0:
            ts = nsigma_clipping(ts, clip)
            ts = local_nsigma_clipping(ts, clip)

    return ts

def process_ptf(ts, filter, clip=None):
    """process ASAS-SN lightcurve data

    Parameters
    ----------

    ts : astropy.Timeseries Object  

    filter : np.int32
        1 for g band, 2 for R band
        
    Returns
    -------

    ts : astropy.Timeseries Object

    """
    ts.sort(['cntr_01', 'time'])

    ts = ts[ts['fid']==filter]

    mask = (ts['mag_auto'] == 'null')
    ts = ts[~mask]

    mask = ((ts['goodflag'] == 1) & (ts['photcalflag']==1))
    ts = ts[mask]
    
    ts['mag_auto'] = ts['mag_auto'].astype(np.float64)
    
    ts = ts[ts['mag_auto']>0]

    ts.keep_columns(['time', 'mag_auto', 'magerr_auto'])

    ts.rename_column('mag_auto', 'mag')
    ts.rename_column('magerr_auto', 'mag_err')

    if clip != None:
        if ts['mag'].shape[0] > 0:
            ts = nsigma_clipping(ts, clip)
            ts = local_nsigma_clipping(ts, clip)

    
    return ts



def bin_data(ts, freq, bins):
    """bin lightcurve data

    Parameters
    ----------

    ts : astropy.Timeseries Object
        timeseries to be binned
    
    freq : np.int32
        size of bins, in days

    epochs : np.int32
        number of bins
        
    Returns
    -------

    binned_ts : astropy.Timeseries Object
        contains number of data points in each bin

    binned_ts_avg : astropy.Timeseries Object
        contains mean of data points in each bin

    binned_ts_var : astropy.Timeseries Object
        contains variance (DOF = 0) of data points in each bin

    ts : astropy.Timeseries Object
        contains original timeseries

    """
    binned_ts = aggregate_downsample(ts, time_bin_size = freq * u.day, n_bins=bins, aggregate_func=np.size)
    binned_ts_avg = aggregate_downsample(ts, time_bin_size = freq * u.day, n_bins=bins, aggregate_func=np.nanmean)
    binned_ts_var = aggregate_downsample(ts, time_bin_size = freq * u.day, n_bins=bins, aggregate_func=np.nanvar)
    
    binned_ts_avg['mag_err'] /= np.sqrt(binned_ts['mag_err'])
    return binned_ts, binned_ts_avg, binned_ts_var, ts



def generate_binned_lc(ts, freq, epochs):
    """bin lightcurve data

    Parameters
    ----------

    ts : astropy.Timeseries Object
        timeseries to be used to generate binned lightcurve (must belong to the same survey)
    
    freq : np.int32
        size of bins, in days

    epochs : np.int32
        number of bins
        
    Returns
    -------

    date : np.float64
        date of each observation

    mag : np.float64
        magnitude observed

    err : np.float64
        estimated error in mag observation
    
    date_avg : np.float64
        date of observation in each bin

    mag_avg : np.float64
        average magnitude observed in each bin

    err_avg : np.float64
        average error in the mean of the mag observed

    """
    date = ts['time'].to_value('decimalyear')
    mag = ts['mag']
    err = ts['mag_err']

    a, b, c, d = bin_data(ts, freq, epochs)
    date_avg = b['time_bin_start'].to_value('decimalyear')
    mag_avg = b['mag']
    err_avg = b['mag_err']/np.sqrt(a['mag_err']) #divide by sqrt(n) for error in the mean

    return date, mag, err, date_avg, mag_avg, err_avg


#def combine_data(data_list):

 #   return combined_lc