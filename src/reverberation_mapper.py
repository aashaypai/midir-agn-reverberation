import astropy.time
import astropy.timeseries
import numpy as np
import astropy
from scipy import optimize

import fileIO_processing as fp
import lightcurve_processing as lp


class FixedWidthModel:
        
    def __init__(self, plateifu, optical_data=None, optical_GP=None, verbose=False, optical_data_mode='raw'):
        
        self._verbose = verbose
        self.plateifu = plateifu

        # create an array that stores the results of every chi-square run
        self.results_log = []

        if optical_data_mode not in ['gp', 'raw']:
            raise ValueError("optical_data_mode must be either 'gp' or 'raw'")
        
        if optical_data_mode=='raw':
            if optical_data is None:
                try:
                    self.optical_data = self.generate_optical_lightcurve()
                    self._assign_optical_ts(self.optical_data)
                except Exception as e:
                    print(f'*** Unable to generate optical data: {e} ***')
            else:
                self._validate_ts(optical_data, 'optical_data')
                self.optical_data = optical_data
                self._assign_optical_ts(self.optical_data)

            if self._verbose: print('*** Using raw optical data ***')
        
        if optical_data_mode=='gp':
            if optical_GP is None:
                try:
                    self.optical_GP = self.generate_optical_gp()
                    self._assign_optical_ts(self.optical_GP)
                except Exception as e:
                    print(f'*** Unable to generate optical GP: {e} ***')
            else:
                self._validate_ts(optical_GP, 'optical_GP')
                self.optical_GP = optical_GP
                self._assign_optical_ts(self.optical_GP)
                
            if self._verbose: print('*** Using optical GP data ***')
    
        # generate W1 & W2 data
        try:
            self.w1, self.w2 = self.generate_wise_lightcurve()
            self._assign_wise_ts()

        except Exception as e:
            print(f'*** Unable to generate IR data: {e} ***')


    def _validate_ts(self, ts, name):
        if not isinstance(ts, astropy.timeseries.TimeSeries):
                raise TypeError(f"{name} must be an astropy.timeseries.TimeSeries object")
        required = {'time', 'mag', 'mag_err'}
        if not required.issubset(ts.colnames):
            raise ValueError(f"{name} must contain columns {required}")
        
    def _assign_optical_ts(self, ts):
        self.t_opt = ts['time'].to_value('decimalyear')
        self.m_opt = ts['mag']
        self.err_opt = ts['mag_err']

    def _assign_wise_ts(self):
        self.t_w1 = self.w1['time'].to_value('decimalyear')
        self.m_w1 = self.w1['mag']
        self.err_w1 = self.w1['mag_err']

        self.t_w2 = self.w2['time'].to_value('decimalyear')
        self.m_w2 = self.w2['mag']
        self.err_w2 = self.w2['mag_err']

    def generate_wise_lightcurve(self): 
        """function that generates the WISE Infrared lightcurve GP for the given Plate-IFU.

        Returns
        -------
        w1 : astropy.timeseries.TimeSeries
            contains WISE W1 data for the given Plate-IFU.

        w2 : astropy.timeseries.TimeSeries
            contains WISE W2 data for the given Plate-IFU.
        """

        mnsa_hdu, manga_wise_hdu, pipe3d_hdu = fp.import_manga(6, 1, 1)
        mnsa, mwv, pipe3d = mnsa_hdu.data, manga_wise_hdu.data, pipe3d_hdu.data 

        w1 = fp.process_wise(self.plateifu, mwv, band=1)
        w2 = fp.process_wise(self.plateifu, mwv, band=2)

        if self._verbose: print('*** W1 & W2 data generated ***')

        return w1, w2
    
    def generate_optical_lightcurve(self):
        """function that generates the optical combined lightcurve for the given Plate-IFU.

        Returns
        -------
        optical_lightcurve : astropy.timeseries.TimeSeries
            contains the combined optical lightcurve data for the given Plate-IFU.

        """
        optical_lightcurve, _ = lp.generate_combined_lightcurve(pifu=self.plateifu)
        if self._verbose: print(f'*** Optical data generated ***')

        return optical_lightcurve
    
    def generate_optical_gp(self, l=[0.95, 1.05]):
        """function that generates the optical GP for the given Plate-IFU.

        Parameters
        ----------
        l : list of float, optional
            the length hyperparameter lower and upper bounds for the GP fit. 
            Default is [0.95, 1.05].
        Returns
        -------
        gp : astropy.timeseries.TimeSeries
            contains a Gaussian Process interpolation of the combined lightcurve for the given Plate-IFU.
        """
        optical_lightcurve, _ = lp.generate_combined_lightcurve(pifu=self.plateifu)

        poly_subtracted_obj_p, fit, fitted_poly = lp.polyfit_lightcurves(optical_lightcurve, deg=10)
        gp, llh, hyperparams, cov = lp.GP(poly_subtracted_obj_p, kernel_num=3, lengthscale=(l[0], l[1]))
        gp_fitted_poly =lp.make_polynomial(gp, fit)
        
        gp['mag']+=gp_fitted_poly
        
        if self._verbose: print(f'*** Optical GP generated *** \n*** GP kernel {str(hyperparams)} ***')

        return gp
    

    def hat(self, width):
        """function that generates the top-hat convolution kernel used to convolve with the optical lightcurve.

        Parameters
        ----------
        width : int
            integer width of the returned array, which is also the width of the kernel
        Returns
        -------
        kern : np.ndarray
            contains the top-hat convolution kernel.
        """
        # cannot make an array of negative length, something has gone wrong.
        if width<0:
            raise ValueError(f"Kernel width cannot be: {width}")
        # width of 0 is equivalent to no convolution, return 1.
        elif width==0:
            return np.ones(1)
        # create an array of ones with a positive width, and then normalize
        # by dividing by the length of the array to get the top-hat kernel 
        y = np.ones(width)
        kern = y/width
        return kern
    
    def convolve(self, lag, width):
        """function that convolves the optical GP with the top-hat kernel 
        to generates the predicted infrared lightcurve.

        Parameters
        ----------
        lag : float
            the time lag between the optical and the IR lightcurve in decimal years

        width : float
            width of the convolution kernel in relation to lag. Eg. if the width is 
            passed to be 0.5, then the kernel length will be 0.5*lag.
        Returns
        -------
        kern : np.ndarray
            contains the top-hat convolution kernel.
        """
        # converts the decimal year width of the kernel to an integer value
        if width*np.abs(lag)<=1: #set maximum smoothing kernel width
            kern_width = int(width*np.abs(lag)/np.nanmedian(np.diff(self.t_opt)))
        else:
            kern_width = int(1/np.nanmedian(np.diff(self.t_opt)))
        kern = self.hat(kern_width)

        # performs the convolution with the kernel and the median subtracted optical GP.
        conv = np.convolve(kern, self.m_opt-np.nanmedian(self.m_opt), mode='valid')
        
        # only keeps the times where the convolution kernel overlaps with the data fully
        if kern_width % 2 == 1:
            pad = (kern_width) // 2

            t_conv = self.t_opt[pad : -pad] if pad > 0 else self.t_opt.copy()
            err_conv = self.err_opt[pad : -pad] if pad > 0 else self.err_opt.copy()

        else:
            pad_left = (kern_width) // 2
            pad_right = kern_width - pad_left - 1#(kern_width-1) // 2

            t_conv = self.t_opt[pad_left : -pad_right] if pad_right > 0 else self.t_opt[pad_left:]
            err_conv = self.err_opt[pad_left : -pad_right] if pad_right >  0 else self.err_opt[pad_left:]

        return conv, t_conv, err_conv
    
    def fit_linear(self, conv, IR_data):
        """function that fits the linear parts (amplitude and constant offset) of the fit
        using least squares minimization.

        Parameters
        ----------
        conv : np.ndarray
            the convolved optical lightcurve

        IR_data : np.ndarray
            the IR data that the optical lightcurve is fitted to
        Returns
        -------
        amp : float
            amplitude of the fit
                
        const : float
            constant offset
        """
        def offset(x):
            a, c = x[0], x[1]
            conv_new = a * conv + c

            difference = conv_new - IR_data
            return difference
        
        amp, const = optimize.leastsq(offset, x0=[1, -50])

        return amp, const
        
    def predict_mags(self, params, IR_data):
        """function that generates the predicted IR magnitudes after convolving and performing a linear least squares
        fit on the optical GP.

        Parameters
        ----------
        params : list of float
            contains the time lag between the IR and optical lightcurves.

        IR_data : astropy.timeseries.TimeSeries
            the IR data that the optical lightcurve is fitted to.
        Returns
        -------
        predicted_mags : np.ndarray
            contains the predicted IR magnitudes 
        
        predicted_errs : np.ndarray
            contains the errors on each predicted IR magnitude

        amp : float
            amplitude of the fit
                
        const : float
            constant offset
        """
        lag = np.round(params[0], 7)#width=, params[1]
        width = 0.5 #self.width
        #width = self.check_width(lag, width)

        conv, t_conv, err_conv = self.convolve(lag, width)
        
        # At this point, you have the IR data from WISE, and the convolved optical data. I have ~20 time values in 
        # the IR data and around ~10000 times in the optical GP. We add the lag to the optical GP to get t_lagged.
        # Now, since the times are discrete points and not continuous, we aren't guaranteed to have the exact IR time values
        # in the optical lagged times (i.e. the IR_data time may contain 2010.12 but the closest value in t_lagged could be 2010.13). 
        # So, we need to find the times in t_lagged that are closest to the IR_data times.
        # If we find the indices of these closest times, we can get the magnitudes that correspond to these times from conv.
        t_lagged = t_conv+lag
        # this code creates a 2D array of the absolute value of the differences between every pair of times in IR_data and t_lagged. 
        diff = np.abs(IR_data['time'].to_value('decimalyear')[:, None] - t_lagged[None, :])

        # finds the indices along axis 1 (the t_lagged axis) with the smallest difference in each row
        inds = np.argmin(diff, axis=1)
        
        # fit the linear amplitude and constant offset
        amp, const = self.fit_linear(conv[inds], IR_data['mag'])[0]
        model = amp * conv + const

        # get the predicted magnitudes after: convolution with a top-hat kernel of specified lag and width, 
        # multiplied by linear amplitude and added constant offset
        # at this point, we have our final prediction for the IR lightcurve based on the optical lightcurve
        predicted_times = t_lagged[inds]
        predicted_mags = model[inds]
        predicted_errs = err_conv[inds]

        #compute weight function to weight chi-square
        W = self.weight_function(conv, inds)
        return predicted_mags, predicted_errs, predicted_times, amp, const, W
    
    def weight_function(self, conv, inds, buffer=1):
        """function that weights the chi-square

        Parameters
        ----------
        conv : np.ndarray
            contains the convolved optical magnitudes

        inds : np.ndarray
            contains the indices of closest overlap of lagged times and IR times

        buffer : int, optional
            the buffer from zeroth and last element to weigh. For example, if the buffer is 1,
            the weight will change only if the index is the 0th or last element. If the buffer
            is 100, any index that is less than 100 away from the zero or that is less than 100 away from
            the last element will be weighted.
        Returns
        -------
        W : float
            weight to assign to the chi-square value.
        """
        # find the number of elements that are less than the buffer value away from the size of the array
        last_element_count = np.size(inds[inds>=np.size(conv)-buffer])
        # find the number of elements that are less than the buffer value away from 0
        first_element_count = np.size(inds[inds<=buffer-1])

        # compute the weight = (no. of overlapping indices/total indices)^2
        W = np.square((np.size(inds)-first_element_count-last_element_count)/np.size(inds))

        return W

    def chisq(self, params, args):
        """function that computes the chi-square value used for fitting.

        Parameters
        ----------
        params : list of float
            contains the time lag between the IR and optical lightcurves.

        IR_data : astropy.timeseries.TimeSeries
            the IR data that the optical lightcurve is fitted to.
        Returns
        -------
        chisq : float
            chi-square value of the fit between the IR_data and predicted IR values
            generated from the optical lighcurve
        """
        # generate IR prediction
        IR_data = args[0]
        model_mags, model_errs, model_times, amp, const, W = self.predict_mags(params=params, IR_data=IR_data)

        chisq = np.sum(((model_mags-IR_data['mag'])/model_errs)**2)

        if self.weighted:
            chisq/=W

        if self._verbose: print(f'lag: {params[0]:.3f}, amp: {amp:.3f}, const: {const:.3f}, chi-square: {chisq:.3f}')
        self.results_log.append([params[0].copy(), amp.copy(), const.copy(), chisq.copy()])
        return chisq
    
    def minimize_chisq(self, IR_data, **kwargs):
        """function that computes the chi-square value used for fitting.

        Parameters
        ----------
        IR_data : int or astropy.timeseries.TimeSeries
        Specifies the infrared dataset to use in the minimization.

        - If set to 1, the function will internally load W1 data.
        - If set to 2, the function will internally load W2 data.
        - If provided as an `astropy.timeseries.TimeSeries` object, it must contain
          the following columns:
            - 'time'     : observation times (Time or Quantity)
            - 'mag'      : magnitudes
            - 'mag_err'  : magnitude uncertainties 

        kwargs : dict, optional
            optional keyword arguments. supported keys:

            - 'ranges' : tuple, default: ((-5, 5, 0.01),)
                    the range over which to brute force fit 
                
            - 'verbose' : bool, default: False
                    verbose option to print out steps and fitting parameters
                
            - 'weighted' : bool, default: False
                    If true, weights chi-square value by overlap between predicted IR points and 
                    IR_data. Discourages lags where the lightcurves do not overlap completely.

        Returns
        -------
        best_fit_params : list of float
            contains the best fit parameters in the following order: lag, amp, const, chi-square
        """
        # If IR_data is 1 or 2, load in W1/W2 data
        match IR_data:
            case 1:
                IR_data = self.w1
                if self._verbose: print(f'*** fitting optical and W1 lightcurves ***')
            case 2:
                IR_data = self.w2
                if self._verbose: print(f'*** fitting optical and W2 lightcurves ***')
            
            # If IR_data is any other lightcurve, it must be an astropy timeseries
            case ts if isinstance(ts, astropy.timeseries.TimeSeries):
                # IR_data must contain the columns "time", "mag" and "mag_err"
                required_cols = {'time', 'mag', 'mag_err'}
                if not required_cols.issubset(ts.colnames):
                    missing = required_cols - set(ts.colnames)
                    raise ValueError(f"TimeSeries is missing required column(s): {', '.join(missing)}")

                IR_data = ts
            case _:
                raise ValueError("IR_data must be 1, 2, or an astropy TimeSeries.")

        self.weighted = kwargs.get("weighted", False)
        self.ranges = kwargs.get("ranges", ((-5, 5, 0.01),))
        self._verbose = kwargs.get("verbose", self._verbose)
        
        # create two arrays that contains the lag, amp, const and chi-square at every iteration of the optimization
        self.results_log = []  
        
        model = optimize.brute(self.chisq, ranges=self.ranges, full_output=self._verbose, disp=self._verbose, 
                               finish=None, args=((IR_data,),)) ## finish=optimize.minimize

        # find where the chi-square is minimized in the results_log, as that contains the amp and const as well
        # since model array will only contain the lag
        # find the minimum chi-square and return that row
        ind = np.argmin(np.array(self.results_log)[:, 3])
        best_fit_params = self.results_log[ind]
        return best_fit_params
