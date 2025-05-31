#from midir import w1w2_condition
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import astropy
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries, aggregate_downsample
from astropy.coordinates import match_coordinates_sky as coords


# from fileIO_processing import *
# from lightcurve_processing import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 22})
matplotlib.rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rc('text', usetex=True)

SAVE_FILEPATH = 'C:/Users/paiaa/Documents/Research/Blanton Lab/Midir Variables/Figures/'
IMPORT_FILEPATH ='C:/Users/paiaa/Documents/Research/Blanton Lab/Midir Variables/'


###########################################################################################################################################################
######### FUNCTIONS FOR FORCED PHOTOMETRY LIGHTCURVES ##########
def import_ztf(fn):

    try:
        tb = Table.read(IMPORT_FILEPATH + '/Lightcurves/ZTF/' + fn + '.txt', format='ascii')

    except:
        fp = '/Lightcurves/ZTF/' + fn + '.txt'
        print('the file at address', fp, "couldn't be found")
        return FileNotFoundError
    
    #renaming columns
    col_names = np.array(['index', 'field', 'ccdid', 'qid', 'filter', 'pid', 'infobitssci', 'sciinpseeing', 'scibckgnd', 'scisigpix', 'zpmaginpsci', 'zpmaginpsciunc', 
                        'zpmaginpscirms', 'clrcoeff', 'clrcoeffunc', 'ncalmatches', 'exptime', 'adpctdif1', 'adpctdif2', 'diffmaglim', 'zpdiff', 'programid', 'jd', 
                        'rfid', 'forcediffimflux', 'forcediffimfluxunc', 'forcediffimsnr', 'forcediffimchisq', 'forcediffimfluxap', 'forcediffimfluxuncap', 
                        'forcediffimsnrap', 'aperturecorr', 'dnearestrefsrc', 'nearestrefmag', 'nearestrefmagunc', 'nearestrefchi', 'nearestrefsharp', 'refjdstart', 
                        'refjdend', 'procstatus'])


    for i, col in enumerate(tb.colnames):
        tb.rename_column(col, col_names[i])
    tb.remove_row(0)

    #converting data to timeseries
    ts = TimeSeries(tb, time=Time(tb['jd'], format='jd'))
    ts.remove_column('jd')

    #filtering bad rows
    try:
        ts = ts[ts['infobitssci'] < 33554432]
        ts = ts[ts['scisigpix'].astype(np.float64) <= 25]
        ts = ts[ts['sciinpseeing'].astype(np.float64) <= 4]
    
    except:
        mask = ((ts['infobitssci'] != 'null') & (ts['scisigpix'] != 'null') & (ts['sciinpseeing'] != 'null'))
        ts = ts[mask]

        ts = ts[ts['infobitssci'] < 33554432]
        ts = ts[ts['scisigpix'].astype(np.float64) <= 25]
        ts = ts[ts['sciinpseeing'].astype(np.float64) <= 4]

    #filtering null rows
    mask = ((ts['forcediffimflux'] != 'null') & (ts['zpdiff'] != 'null') & (ts['forcediffimfluxunc'] != 'null'))

    ts = ts[mask]
    
    #converting remaining rows to floats
    ts['forcediffimflux'] = ts['forcediffimflux'].astype(np.float64)
    ts['forcediffimchisq'] = ts['forcediffimchisq'].astype(np.float64)
    ts['forcediffimfluxunc'] = ts['forcediffimfluxunc'] .astype(np.float64)
    ts['zpdiff'] = ts['zpdiff'].astype(np.float64)
    
    return ts


def isolate_ids(ts, filter='g', verbose=True):

    fields = np.unique(ts['field']).data
    ccds = np.unique(ts['ccdid']).data
    qids = np.unique(ts['qid']).data
    filters = np.unique(ts['filter']).data

    field_obs_count, ccd_obs_count, qid_obs_count, filter_obs_count = np.array([]), np.array([]), np.array([]), np.array([]),  
    
    for field in fields:
    
        field_obs_count = np.append(field_obs_count, ts[ts['field']==field]['field'].size)

    for ccd in ccds:

        ccd_obs_count = np.append(ccd_obs_count, ts[ts['ccdid']==ccd]['ccdid'].size)

    for qid in qids:

        qid_obs_count = np.append(qid_obs_count, ts[ts['qid']==qid]['qid'].size)

    for f in filters:

        filter_obs_count = np.append(filter_obs_count, ts[ts['filter']==f]['filter'].size)

    
    selected_field = fields[np.argmax(field_obs_count)]
    selected_ccd = ccds[np.argmax(ccd_obs_count)]
    selected_qid = qids[np.argmax(qid_obs_count)]

    if 'ZTF_' + filter not in filters:
        print('no observations available in the chosen filter:', filter, '\n', 'reverting to filter with the most observations.')
        selected_filter = filters[np.argmax(filter_obs_count)]
    else:
        selected_filter = 'ZTF_' + str(filter)
    
    if verbose==True:
        #prints all possible ids in descending order of number of rows for each id
        print('fields:', fields[np.argsort(field_obs_count)][::-1], 'field sizes:', np.sort(field_obs_count)[::-1])
        print('ccds:', ccds[np.argsort(ccd_obs_count)][::-1], 'ccd sizes:', np.sort(ccd_obs_count)[::-1])
        print('qids:', qids[np.argsort(qid_obs_count)][::-1], 'qid sizes:', np.sort(qid_obs_count)[::-1])
        print('filters:', filters[np.argsort(filter_obs_count)][::-1], 'filter sizes:', np.sort(filter_obs_count)[::-1])

    return selected_field, selected_ccd, selected_qid, selected_filter


def filter_ts(ts, selected_field, selected_ccd, selected_qid, selected_filter):

    mask = ((ts['field'] == selected_field) & (ts['ccdid'] == selected_ccd) & (ts['qid'] == selected_qid) & (ts['filter'] == selected_filter))

    return ts[mask]


#########################################################################################################################################################
######### FUNCTIONS FOR IRSA-QUERIED LIGHTCURVES ##########
def find_oid(plateifu, key, filter='zg'):

    key = key[key['plateifu_01'] == plateifu]
    
    if key[key['filtercode'] == filter]['plateifu_01'].size == 0:
        print('no observations available in the chosen filter:', filter, '\n', 'reverting to filter with the most observations.')
        filter = key['filtercode'][np.argmax(key['ngoodobs'])]
        key = key[key['filtercode'] == filter]

    else:
        key = key[key['filtercode'] == filter]

    
    oid = key['oid'][np.argmax(key['ngoodobs'])]

    return oid

def find_lightcurve(oid, lc):

    lc = lc[lc['oid'] == oid]
    ts = TimeSeries(lc, time=Time(lc['mjd'], format='mjd'))
    return ts