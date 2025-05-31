import numpy as np
import os
from astropy.io import fits

import astropy
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries, aggregate_downsample
import astropy.units as u

from scipy.optimize import minimize

from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

import fileIO_processing as fp
import ZTF_lightcurve_processing as zlp

project_root = os.path.abspath(os.path.join(os.path.dirname('mid-ir-agn-dust-reverberation'), ".."))
IMPORT_FILEPATH = os.path.join(project_root, "lightcurves")
###################################################################################################################################################

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



def local_nsigma_clipping(ts, n, size=5, verbose=False):
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
    
    mask = np.array([])
    for i, obs in enumerate(mag):
        
        if i < size or i > mag.shape[0] - size:
            mask = np.append(mask, False)
            continue

        
        med = np.nanmedian(mag[i-size:i+size])
        
        std = np.std(mag[i-size:i+size])
        
        #print('median:', str(med))
        
        if ((obs < med + n*std) and (obs > med - n*std)):
            time_new = np.append(time_new, time[i])
            mag_new = np.append(mag_new, mag[i])
            mag_err_new = np.append(mag_err_new, mag_err[i])
            mask = np.append(mask, True)
        
        else:
            clipped_count +=1 
            mask = np.append(mask, False)

    time_new = np.append(time_new, time[-1*size:])
    mag_new = np.append(mag_new, mag[-1*size:])
    mag_err_new = np.append(mag_err_new, mag_err[-1*size:])
    
    if verbose==True:
        print('points removed:', str(clipped_count))

    tb_new = Table()
    tb_new['mag'] = mag_new
    tb_new['mag_err'] = mag_err_new

    ts_new = TimeSeries(tb_new, time=Time(time_new, format='decimalyear'))

    ts.add_column(mask, name='mask')
    return ts_new, ts



def GP(ts_in, kernel_num, lengthscale):
    """perform gaussian process regression on input timeseries
    Parameters
    ----------

    ts_in : astropy.Timeseries Object
        timeseries to be used for gaussian process regression
    Returns
    -------

    ts: astropy.Timeseries Object
        timeseries containing gaussian process regression results

    hyper_vector: np.ndarray
        array containing information about hyperparameters
    """
   
    x_in = ts_in['time'].to_value('decimalyear')
    y_in = ts_in['mag']
    y_err = ts_in['mag_err']

    # Define range of input space to predict over
    x_min = x_in.min() 
    x_max = x_in.max() 
    
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x_space = np.atleast_2d(np.linspace(x_min, x_max, 10000)).T #5 * x_in.shape[0]
    x_fit = np.atleast_2d(x_in).T
    
    l = (lengthscale[1]-lengthscale[0])/2
    k_RBF = kernels.RBF(length_scale=l, length_scale_bounds=(lengthscale[0], lengthscale[1]))
    k_exp = (kernels.Matern(length_scale=l, length_scale_bounds=(lengthscale[0], lengthscale[1]), nu=0.5))
    k_sine = kernels.ExpSineSquared(length_scale=l, length_scale_bounds=(lengthscale[0], lengthscale[1]), periodicity=1e1, periodicity_bounds=(1e-2, 1e4))
    k_noise = kernels.WhiteKernel(noise_level=l, noise_level_bounds=(0.01, 1))
    k_matern = (kernels.Matern(length_scale=l, length_scale_bounds=(lengthscale[0], lengthscale[1]), nu=1.5))
    # Matern kernel with nu = 0.5 is equivalent to the exponential kernel
    # Define kernel function
    if kernel_num == 0:
        kernel = 1.0 * k_exp + k_noise#+ k_noise #+ k_RBF + 1.0*(k_exp*k_sine)
    if kernel_num == 1:
        kernel = 1.0 * k_matern #+ k_noise #NOT GOOD, BLOWS UP A LOT
    if kernel_num == 2:
        kernel = 1.0 * k_sine #+ k_noise
    if kernel_num == 3:
        kernel = 1.0 * k_RBF + k_noise
    if kernel_num == 4:
        kernel = (1.0 * k_matern + 1.0 * k_exp ) * 1.0 * k_RBF 
    if kernel_num == 5:
        kernel =  1.0 * k_matern + 1.0 * k_exp #+ k_noise

    if kernel_num == 6:
        kernel = 1.0 * k_RBF + 1.0 * k_noise
        
    
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=y_err**2, n_restarts_optimizer=10, normalize_y=True, random_state=1)
    
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gpr.fit(x_fit, y_in)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, y_pred_cov = gpr.predict(x_space, return_cov=True)

    y_pred_sigma = np.sqrt(np.diag(y_pred_cov))
    # Get log likelihood and hyperparameters
    log_likelihood = gpr.log_marginal_likelihood()
    hyper_params = gpr.kernel_
    
    # ts_avg = aggregate_downsample(ts_in, time_bin_size = 30 * u.day, n_bins=400, aggregate_func=np.nanmean)
    # ts_avg = ts_avg[~ts_avg['mag'].mask]
    # ts_avg['time'] = ts_avg['time_bin_start']

    # cv = LeaveOneOut()
    # scores = cross_val_score(gpr, ts_avg['time'].to_value('decimalyear').reshape(-1, 1), ts_avg['mag'], scoring='neg_mean_absolute_error',cv=cv, n_jobs=-1)
    # RMSE = (np.sqrt(np.mean(np.absolute(scores))))
    # hyper_vector = []
    # hyper_vector.append(log_likelihood)
    # params = hyper_params.get_params()
    # for i, key in enumerate(sorted(params)):
    #     if i in (3,6,10,14,18,20,23):
    #         #print(i, "%s : %s" % (key, params[key]))
    #         hyper_vector.append(params[key])
    
    #compile data into Timeseries
    RMSE=0
    x_space = x_space.flatten()
    y_pred = y_pred.flatten()
    y_pred_sigma = y_pred_sigma.flatten()

    tb = Table()

    tb['mag'] = y_pred
    tb['mag_err'] = y_pred_sigma

    ts = TimeSeries(tb, time=Time(x_space, format='decimalyear'))

    return ts, log_likelihood, hyper_params, y_pred_cov


def match_lightcurves(ts1, ts2):
    """find constant offset between two overlapping regions of two light curves that minimizes the difference between the two
    Parameters
    ----------

    ts1 : astropy.Timeseries Object
        light curve that begins and ends first

    ts2 : astropy.Timeseries Object
        light curve that begins and ends later   
    Returns
    -------

    const: np.float64
        constant offset between the two light curves
    """

    t1 = ts1['time'].to_value('decimalyear')
    m1 = ts1['mag']

    t2 = ts2['time'].to_value('decimalyear')
    m2 = ts2['mag']

    t_diff_start = t1 - t2[0]
    t_diff_end = t2 - t1[-1]

    zero_crossing_start = np.where(np.diff(np.signbit(t_diff_start)))[0]+1
    zero_crossing_end = np.where(np.diff(np.signbit(t_diff_end)))[0]

    if zero_crossing_start.shape[0] == 0 or zero_crossing_end.shape[0] == 0:
        print('using padded times')
        time_diff = (t2[0] - t1[-1])+0.1


        padded_t1 = np.pad(t1, pad_width=(0, 10), mode='linear_ramp', end_values=t1[-1] + time_diff)
        # print(t1[-1] + time_diff)
        padded_t2 = np.pad(t2, pad_width=(10, 0), mode='linear_ramp', end_values=t2[0] - time_diff)
        # print(t2[0] - time_diff)

        t_diff_start = padded_t1 - padded_t2[0]
        t_diff_end = padded_t2 - padded_t1[-1]

        zero_crossing_start = np.where(np.diff(np.signbit(t_diff_start)))[0]+1
        zero_crossing_end = np.where(np.diff(np.signbit(t_diff_end)))[0]

    
    overlap1 = m1[zero_crossing_start[0]:]     
    if zero_crossing_end[0] != 0:              ##edge case where the zero crossing happens at the first element of the second timeseries
        overlap2 = m2[:zero_crossing_end[0]]   
    else:
        overlap2 = m2[:zero_crossing_end[0]+1] ##add 1 because python does not include the nth element in array[:n], here n=0.

    if overlap1.shape[0] < overlap2.shape[0]:
        zero_crossing_end -= (overlap2.shape[0] - overlap1.shape[0])

    if overlap1.shape[0] > overlap2.shape[0]:
        zero_crossing_start += (overlap1.shape[0] - overlap2.shape[0])

    overlap1 = m1[zero_crossing_start[0]:]

    if zero_crossing_end[0] != 0:              ##edge case where the zero crossing happens at the first element of the second timeseries
        overlap2 = m2[:zero_crossing_end[0]]
    else:
        overlap2 = m2[:zero_crossing_end[0]+1] ##add 1 because python does not include the nth element in array[:n], here n=0.

    def minimize_func(c):
        v1, v2 = overlap1, overlap2
        shifted_v2 = v2+c

        return np.sqrt(np.sum(np.square(v1-shifted_v2)))
    
    const = minimize(minimize_func, x0=0).x

    return const

def make_polynomial(ts, fit):
    """make an array containing the y values of a polynomial function using the coefficients provided
    ----------

    ts : astropy.Timeseries Object
        light curve timeseries that contains the x values (time)

    fit: np.array
        contains the coefficients of the polynomial of the form f(x) = a*x^n + b*x^(n-1) + ... + const.   
    Returns
    -------
    fitted_poly : np.array
        array containing y values of the polynomial f(x)
    """
    fitted_poly = np.zeros(ts['time'].to_value('decimalyear').shape[0])
    for i in range(fit.shape[0]):
        fitted_poly+= fit[::-1][i] * ts['time'].to_value('decimalyear') ** (i)

    return fitted_poly

def polyfit_lightcurves(ts, deg):
    """fit a polynomial of degree deg to a lightcurve and subtract the polynomial to obtain the residuals of the fit
    ----------

    ts : astropy.Timeseries Object
        light curve to which the polynomial will be fit

    deg : np.int32
        degree of the polynomial
  
    Returns
    -------
    new_ts : astropy.Timeseries Object
        timeseries containing the magnitude values after the fitted polynomial has been subtracted
    
    fit : np.array
        array containing the coefficients of the polynomial of the form f(x) = a*x^n + b*x^(n-1) + ... + const.  

    fitted_poly : np.array
        array containing y values of the polynomial f(x)
    """   
    fit = np.polyfit(ts['time'].to_value('decimalyear'), ts['mag'], deg=deg)

    fitted_poly = make_polynomial(ts, fit)

    new_ts = Table()
    new_ts['mag'] = ts['mag'] - fitted_poly
    new_ts['mag_err'] = ts['mag_err']
    new_ts = TimeSeries(new_ts, time=ts['time'])

    return new_ts, fit, fitted_poly


def average_mag_around_specific_times(ts1, ts2, size=5):
    """find the closest x (time) value matches from ts1 in ts2 and average the magnitude around those x values
    Parameters
    ----------

    ts1 : astropy.Timeseries Object
        light curve that has more data points 

    ts2 : astropy.Timeseries Object
        light curve that has less data points   

    size : np.int32
        number of data points to use to take the average 
    Returns
    -------

    t: astropy.Timeseries Object
        timeseries containing the averaged magnitude values around the closest matches of time values from ts1 in ts2
    """
    t1 = ts1['time'].to_value('decimalyear')
    t2 = ts2['time'].to_value('decimalyear')

    inds = np.abs(t1[:, None] - t2[None, :]).argmin(axis=-1)
    #print(inds.shape)
    #print(inds)
    times = np.array([])
    avg = np.array([])
    avg_err = np.array([])

    for i in inds:
        times = np.append(times, ts2['time'][i].to_value('decimalyear'))
        if (i > size) and (i < ts2['mag'].shape[0] - size):
            avg = np.append(avg, np.nanmean(ts2['mag'][i-size:i+size]))
            avg_err = np.append(avg_err, np.nanmean(ts2['mag_err'][i-size:i+size])/np.sqrt(ts2['mag_err'][i-size:i+size].shape[0]))
        else:
            #print(ts2['mag'][i])
            avg = np.append(avg, ts2['mag'][i])
            avg_err = np.append(avg_err, ts2['mag_err'][i])

    t = Table()
    t['mag'] = avg
    t['mag_err'] = avg_err
    t = TimeSeries(t, time=Time(times, format='decimalyear'))
    return t


def generate_combined_lightcurve(pifu):

    crts1 = TimeSeries.read(IMPORT_FILEPATH + '/CRTS/' + 'crts1.csv', format='csv', time_column='MJD', time_format='mjd')
    crts2 = TimeSeries.read(IMPORT_FILEPATH + '/CRTS/' + 'crts2.csv', format='csv', time_column='MJD', time_format='mjd')
    crts = astropy.table.vstack([crts1, crts2])

    ptf = TimeSeries.read(IMPORT_FILEPATH + '/PTF/' + 'PTF_midir_variables.csv', format='csv', time_column='obsmjd', time_format='mjd')
    #asassn = TimeSeries.read(IMPORT_FILEPATH + '/Lightcurves/ASAS-SN/' + asassn_name, format='csv', time_column='HJD', time_format='jd')

    ztf = Table.read(IMPORT_FILEPATH + '/ZTF/' + 'ztf_lc_key.tbl', format='ascii.ipac')
    ztf_lc = fits.open(IMPORT_FILEPATH + '/ZTF/' + 'lc.fits')[1].data

    crts_obj = crts[crts['InputID'] == pifu]
    ptf_obj = ptf[ptf['plateifu_01'] == pifu]
    asassn_obj = TimeSeries.read(IMPORT_FILEPATH + '/ASAS-SN/' + pifu +'.csv', format='ascii', time_column='JD', time_format='jd')

    ztf_oid = zlp.find_oid(pifu, ztf)
    ztf_obj = zlp.find_lightcurve(ztf_oid, ztf_lc)



    crts_obj_p = fp.process_crts(crts_obj, clip=2)
    ptf_obj_p = fp.process_ptf(ptf_obj, 2, clip=2)
    asassn_obj_p = fp.process_asassn(asassn_obj, 'V', clip=2)
    ztf_obj_p = fp.process_ztf(ztf_obj, filter='zg', clip=2)



    if np.size(crts_obj_p)> 0:
        crts_a, crts_b, crts_c, crts_d = fp.bin_data(crts_obj_p, freq=20, bins=200)
        crts_b = crts_b[~crts_b['mag'].mask]
        crts_b['time'] = crts_b['time_bin_start']
        crts_final_date = crts_obj_p['time'][-1]
    else:
        crts_b = np.array([])
        crts_final_date = Time(0, format='decimalyear')

    if np.size(asassn_obj_p)> 0:    
        asassn_a, asassn_b, asassn_c, asassn_d = fp.bin_data(asassn_obj_p, freq=20, bins=200)
        asassn_b = asassn_b[~asassn_b['mag'].mask]
        asassn_b['time'] = asassn_b['time_bin_start']
        asassn_final_date = asassn_obj_p['time'][-1]
        if np.size(crts_b)> 0 and np.size(asassn_b) > 0:
            const1 = match_lightcurves(crts_b, asassn_b)
        else:
            const1=0        

    
        asassn_b['mag'] += const1
    
    else:
        asassn_b = np.array([])
        asassn_final_date=Time(0, format='decimalyear')

    if np.size(ztf_obj_p) > 0:
        ztf_a, ztf_b, ztf_c, ztf_d = fp.bin_data(ztf_obj_p, freq=20, bins=150)
        ztf_b = ztf_b[~ztf_b['mag'].mask]
        ztf_b['time'] = ztf_b['time_bin_start']
        ztf_final_date=ztf_obj_p['time'][-1]

        if np.size(asassn_b) > 0 and np.size(ztf_b) > 0:
            const2 = match_lightcurves(asassn_b, ztf_b)
        else:
            const2=0
    else:
        ztf_b = np.array([])
        ztf_final_date=Time(0, format='decimalyear')

    asassn_obj_p['mag'] += const1
    ztf_obj_p['mag'] += const2

    combined_obj = astropy.table.vstack([crts_obj_p, asassn_obj_p, ztf_obj_p])
    survey = np.array([])
    survey = np.append(survey, np.repeat('crts', np.size(crts_obj_p)))
    survey = np.append(survey, np.repeat('asassn', np.size(asassn_obj_p)))
    survey = np.append(survey, np.repeat('ztf', np.size(ztf_obj_p)))
    combined_obj.add_column(survey, name='survey')
    #combined_obj.sort(['time'])
    _, combined_obj_p = local_nsigma_clipping(combined_obj, 2)

    combined_obj_p = combined_obj_p[combined_obj_p['mask']==True]

    sizes=np.ones(3)
    last_datapoint_time = (crts_final_date, asassn_final_date, ztf_final_date)

    sizes[0] = (np.size(combined_obj_p[combined_obj_p['time']<=last_datapoint_time[0]]))
    sizes[1] = (np.size(combined_obj_p[(last_datapoint_time[0]<combined_obj_p['time']) & (combined_obj_p['time']<=last_datapoint_time[1])]))
    sizes[2] = (np.size(combined_obj_p[(last_datapoint_time[1]<combined_obj_p['time']) & (combined_obj_p['time']<=last_datapoint_time[2])]))

    return combined_obj_p, sizes

