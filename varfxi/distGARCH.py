# -*- coding: utf-8 -*-
"""
Distributional GARCH package
Wrapper around the excellent ARCH package by K.Sheppard 
https://arch.readthedocs.io/
Romain Lafarguette 2020, rlafarguette@imf.org
Time-stamp: "2021-09-24 22:48:41 RLafarguette"
"""

###############################################################################
#%% Modules import
###############################################################################
# Standard imports
import os, sys, importlib                               # Operating system
import pandas as pd                                     # Dataframes
import datetime                                         # Dates management
import numpy as np                                      # Numeric package
import scipy                                            # Scientific package
import arch                                             # ARCH/GARCH models
import itertools

# ARCH package functional imports
from arch.univariate import ARX, ZeroMean # Drift model
from arch.univariate import ConstantVariance, FixedVariance, ARCH, EGARCH, GARCH, GARCH, EWMAVariance, RiskMetrics2006 # Volatility
from arch.univariate import Normal, StudentsT, SkewStudent, GeneralizedError # Distributions

# Other functional imports
from collections import namedtuple                      # Named tuples
from scipy import stats                                 # Statistical tools 
from statsmodels.distributions.empirical_distribution import ECDF

# Graphics
# NB: for some systems, need to remove matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm                               # Colormaps
import matplotlib.colors as mcolors
import seaborn as sns                                   # Graphical package
from tqdm import tqdm

# Local modules
from varfxi.joyplot2 import joyplot2

###############################################################################
#%% Ancillary functions
###############################################################################
# Format floating numbers
def formatFloat(num):
    if num % 1 == 0: # Convert into an integer if possible
        return(int(num))
    else: # Keep as it is
        return(num) 

# Flatten nested lists
def flatten(nested_l):
    return([item for sublist in nested_l for item in sublist])

# Find the closest element in a list, return index and value
def closest(value, lst):
    """ Find the closest element in a list and return (index, value) """
    lst = np.asarray(lst) 
    idx = (np.abs(lst - value)).argmin() 
    return(idx, lst[idx])

# Return the next business bay (or later) for any date
def next_business_day(timestamp, horizon=1):
    """ Return the next business bay (or later) for any date """
    # Pay attention: days=0 means next business day (offset 1 by default)
    timestamp = pd.to_datetime(timestamp)
    
    bd = pd.tseries.offsets.BusinessDay(
        offset=datetime.timedelta(days=horizon-1))
    next_day = timestamp + bd
    return(next_day)

# Avoid jumps in plots
# https://stackoverflow.com/questions/1273472/how-to-skip-empty-dates-weekends-in-a-financial-matplotlib-python-graph/2335781
from matplotlib.ticker import FuncFormatter

def equidate_ax(fig, ax, dates, fmt="%Y-%m-%d", label="Date"):
    """
    Sets all relevant parameters for an equidistant date-x-axis.
    Tick Locators are not affected (set automatically)

    Args:
        fig: pyplot.figure instance
        ax: pyplot.axis instance (target axis)
        dates: iterable of datetime.date or datetime.datetime instances
        fmt: Display format of dates
        label: x-axis label
    Returns:
        None

    """    
    N = len(dates)
    def format_date(index, pos):
        index = np.clip(int(index + 0.5), 0, N - 1)
        return dates[index].strftime(fmt)
    ax.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax.set_xlabel(label)
    fig.autofmt_xdate()

###############################################################################
#%% Class: DistGARCH
###############################################################################
class DistGARCH(object):
    """ 
    Estimate a Distribution Model based on GARCH
    With different specifications for drift, volatility and errors distribution
    Based on the excellent ARCH Python package produced by K.Sheppard

    Inputs
    ------
    depvar_str: str
       Name of the dependent variable

    data = pandas dataframe
       Historical data to fit the model on 

    level_str: str
       Name of the dependent variable in level (in case depvar is returns)

    exog_l: list; default None
        List of exogeneous variables for the ARX process

    lags_l: list; default None
        List of lags for the AR(X) process
        
    vol_model: arch volatility model, default GARCH(1,1)
        Volatility model from the arch package. Possible models include:
        ARCH, GARCH, EGARCH, EWMA, etc.
        https://arch.readthedocs.io/en/latest/univariate/volatility.html

    dist_family: arch distribution family, default Normal()
        Family for the distribution of the errors terms. Can include
        Normal(), SkewStudent(), StudentsT()
        https://arch.readthedocs.io/en/latest/univariate/distribution.html
        
    init_type: str defining the type of initialization, by defualt 'ARX'.
        it takes 3 possible values:
            'ARX': ARX model
                conditional mean = AR + exogenous variable 
                conditional volatility = GARCH Type model
                
            'ZEROMEAN': to estimate the properties of the residuals 
                conditional mean = 0
                conditional volatility = GARCH Type model
                
            'FIXED': Recreate the model from an already fitted model
        
    params: pd.Series(). default empty
        Model parameters, if init_type ='FIXED'
    

    Output
    ------
    A GARCH class object, wrapping .fit() and .fit().forecast() classes

    Usage:
    dg = DistGARCH(depvar_str, level_str, data)

    """
    __description = "Distributional GARCH"
    __author = "Romain Lafarguette - romainlafarguette@github.io, Amine Raboun - amineraboun@github.io"

    # Initializer
    def __init__(self, 
                 depvar_str,
                 data,
                 level_str=None, 
                 exog_l= None,
                 lags_l= None,
                 vol_model=GARCH(1,1),
                 dist_family=Normal(),
                 init_type = 'ARX',
                 params = None,
                 random_state=42, # Answer to the Ultimate Question of Life 
    ):
        
        # Special attributes
        self.vol_model = vol_model # By default GARCH(1,1)
        
        self.dist_family = dist_family # By default Normal distribution
        self.rs = np.random.RandomState(random_state) 
        self.dist_family._random_state = self.rs
        self.init_type = init_type
            
        if init_type in ['ARX', 'FIXED']:
            if init_type=='FIXED':
                assert isinstance(params, pd.Series), 'params must be structured in a pandas Series'
                assert not params.empty, 'params empty'
                self.params = params
                
            # Attributes
            self.depvar = depvar_str
            self.level = level_str
            self.exog_l = exog_l
            
            #import pdb; pdb.set_trace()
            if isinstance(lags_l, int):
                if lags_l==0:
                    self.lags_l = None
                else:
                    self.lags_l = np.arange(1, lags_l+1)
            elif isinstance(lags_l, list):
                if 0 in lags_l:
                    lags_l.remove(0)
                if len(lags_l) > 0:                  
                        self.lags_l = lags_l
                else:
                    self.lags_l = None
            else:
                self.lags_l = None

            #### Data treatment
            # Aligning all the variables, take care of possible None        
            v_l = [self.depvar]
            if self.level:
                v_l = [self.depvar, self.level]
            if self.exog_l: # Accomodate missing level too
                v_l = v_l + self.exog_l          
            self.df = data.loc[:, v_l].dropna().copy() # align all variables 

            # Subselect exogeneous variables if possible
            if self.exog_l:
                self.df_exog = self.df.loc[:, list(self.exog_l)].copy()
            else:
                self.df_exog = None # Keep None

            #### Main Models    
            # Mean (drift) model, potentially with exogeneous regressions and lags
            self.mod = ARX(y=self.df[self.depvar], 
                           x=self.df_exog,
                           constant=True, # Fit intercept by defaults
                           lags=self.lags_l, 
                           volatility=self.vol_model,
                           distribution=self.dist_family)
            

        elif init_type=='ZEROMEAN':
            #### Main Models    
            # Mean (drift) model, potentially with exogeneous regressions and lags
            if isinstance(data, pd.Series):
                self.depvar = 'residuals'
                self.df = data.dropna().rename(self.depvar).to_frame()
            elif isinstance(data, pd.DataFrame):
                if depvar_str in data.columns:
                    self.depvar = depvar_str
                    self.df = data[self.depvar].dropna().to_frame()
                elif data.shape[1] ==1:
                    self.depvar = 'residuals'
                    self.df = data.squeeze().dropna().rename(self.depvar).to_frame()
                else:
                    raise ValueError("data has a format not adequat to the Zeromean model")
            else:
                raise ValueError("data has a format not adequat to the Zeromean model")
            
            self.level = self.depvar
            self.exog_l = None 
            self.lags_l = None
            self.df_exog = None
            self.mod = ZeroMean(y=self.df[self.depvar],
                                volatility=self.vol_model,
                                distribution=self.dist_family)            
        else:
            raise ValueError('Unrecognized initialization type.')
        
        # Plots
        self.plot = self.__plot()
        
    # Class-methods (methods which returns a class defined below)    
    def fit(self,
            cov_type='robust',
            disp='off',
            update_freq=1,
            verbose = True
            ):
        if self.init_type=='FIXED':            
            return(DistGARCHFit(self, cov_type= cov_type, disp=disp, update_freq=update_freq, fixed =True, params =self.params, verbose=verbose))
        else:
            return(DistGARCHFit(self, cov_type= cov_type, disp=disp, update_freq=update_freq, fixed =False, params =None, verbose=verbose))
    
    # Class-methods (methods which returns a class defined below)    
    def optimize(self):
        if self.init_type =='FIXED':
            print('No optimization for Fixed model.')
            return None
        else:
            return(DistGARCHOptimize(self))
        
    def forecast(self, 
                 start_date=None,
                 horizon=1,
                 fmethod='analytic',
                 sample_size=10000
                ):
        
        start_date = self.df.index[-1] if (start_date==None) else start_date
        if self.init_type =='FIXED':
            fitlike = self.fit()            
            return(DistGARCHForecast(fitlike, start_date, horizon, fmethod, sample_size))
        
        else:
            print('Forecst function is only available for for Fixed model Initialization.')
            print('Other initializations need to be fitted first')
            return None
    
    def __plot(self):
        return DistGARCHPlot(self)       
        
###############################################################################
#%% Class: DistGARCHFit
###############################################################################
class DistGARCHFit(object): # Fitted class for the DistGARCH class
    """ 
    Fit a DistGarch model

    General documentation
    https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.ARX.fit.html#arch.univariate.ARX.fit

    Inputs
    ------
    cov_type: str
        Type of covariance
    
    disp: str, 'off' or 'on'
        Display the results of the iterations 

    update_freq: integer, default=1
        Frequency of iteration updates
        Output is generated every update_freq iterations    
        Set to 0 to disable iterative output
    
    """

    # Initialization
    def __init__(self, DistGARCH, cov_type, disp, update_freq, fixed, params, verbose):
   
        self.__dict__.update(DistGARCH.__dict__) # Import all attributes

        if fixed:
            print("Model is fixed and won't be fitted again. The parameter provided will be used")
            assert isinstance(params, pd.Series), 'params must be structured in a pandas Series'
            assert not params.empty, 'params empty'
            self.res = self.mod.fix(params=params)
            
        else:
            # Fit the model (just take from the ARCH model)    
            self.res = self.mod.fit(
                cov_type=cov_type, 
                disp=disp, 
                update_freq=update_freq)

        if verbose:
            # Print the summary at each fit
            print(self.res.summary())

        # Erros Distribution parameters (depends on the distribution)
        # Note that the distributions are NORMALIZED
        dist_names = ['Normal',
                      "Standardized Student's t",
                      "Standardized Skew Student's t",
                      'Generalized Error Distribution']
        
        if self.mod.distribution.name in dist_names:
            dist_params_names = self.mod.distribution.parameter_names()
            self.dist_params = self.res.params.loc[dist_params_names]
                                    
        else:
            raise ValueError('Distribution name mis-specified')


        # Compute the historical and in-sample volatility and mean
        self.hist_avg = self.df[self.depvar].mean()
        self.hist_vol = self.df[self.depvar].std()
        if not fixed:
            self.in_cond_vol = self.res.conditional_volatility
        
        # Plots
        self.plot = self.__plot()
        
        return None
        
    # Class-methods (methods which returns a class defined below)    
    def forecast(self, 
                 start_date=None,
                 horizon=1,
                 fmethod='analytic',
                 sample_size=10000
                ):
        start_date = self.df.index[-1] if (start_date==None) else start_date
        return(DistGARCHForecast(self, start_date, horizon, fmethod,
                                 sample_size))


    # Public methods: Shocks Simulation
    def shock_simulate(self, nobs=100, burn=0, mean_mult=1, vol_mult=1):
        """ 
        Simulate the model based on estimated parameters. Exogenous variables are not be simulated. 
        The simulations assume a constant mean model centered around the historical average

        nobs: integer, default 100
            Number of observations of the simulated frame

        burn: integer, default 0
            Number of initial observations to "burn". Put at 0 to study shocks

        mean_mult: float, default 1
            Multiplier on hist average to adjust the initial shock

        vol_mult: float, default 1
            Multiplier on hist volatility to increase the initial shock

        """
                
        # Extract the parameters without exogeneous variables
        # Exogeneous variables can not be forecasted...
        if self.exog_l:
            spar_index = [x for x in self.res.params.index
                          if x not in self.exog_l]
        else:
            spar_index = self.res.params.index
    
        vparams = self.res.params[spar_index]

        # Simulate the model
        # Linear volatility models (in x)
        # ARCH => initial_value_vol=np.power(x*hist_vol,2) (variance)
        # GARCH => initial_value_vol=np.power(x*hist_vol,2) (variance)
        # EWMA => initial_value_vol=np.power(x*hist_vol,2) (variance)

        # Non-linear volatility models (linear if transform x...)
        # EGARCH => initial_value_vol=np.log((x*hist_vol)**2) (log of variance)

        # For processes linear in variance (not vol)
        if self.mod.volatility.name in ['ARCH', 'GARCH', 'GJR-GARCH',
                                        'EWMA/RiskMetrics',
                                        'RiskMetrics2006']:
            init_vol = np.power(vol_mult*self.hist_vol,2)
        elif self.mod.volatility.name in ['EGARCH']: # Exponential vol
            init_vol = np.log((vol_mult*self.hist_vol)**2)
        else:
            raise ValueError((f'Volatility model {self.mod.volatility.name}'
                             ' not incorporated yet'))

        # Run the simulation
        dsim = self.res.model.simulate(vparams, nobs=nobs, burn=burn,
                                       initial_value= mean_mult*self.hist_avg,
                                       initial_value_vol= init_vol)

        # Provide normalized volatility
        dsim['norm_volatility'] = dsim['volatility']/self.hist_vol
        return(dsim)

    # Public methods: summary table
    def summary_table(self, 
                      model_name='GARCH model',
                      var_d=None,
                      round_param=2,
                      print_pval=True):
        """
        Inputs
        ------

        model_name=str, default "GARCH"
            Name of the model which will appear in the table column

        var_d: dict, default None
            Dictionary for renaming the variables as index

        round_param: int, default 3
            Rounding parameter for the model coefficients

        """

        # Preparation of the table
        st = 'Significance *10%, **5%, ***1%'
        if print_pval==True:
            var_l = list(self.res.params.index) + ['R2', 'R2 adjusted',
                                                   'Number of observations', 
                                                   'Pvalue in parenthesis',
                                                   st]
        else:
            var_l = list(self.res.params.index) + ['R2', 'R2 adjusted',
                                                   'Number of observations', 
                                                   st]


        summary_table = pd.DataFrame(index=var_l, columns=[model_name])

        # Populate the table
        for variable in self.res.params.index:
            param = self.res.params[variable]
            pval = self.res.pvalues[variable]

            if (pval<=0.1) & (pval>0.05):
                stars='*'
            elif (pval<=0.05) & (pval>0.01):
                stars='**'
            elif pval<=0.01:
                stars='***'
            else:
                stars=''

            if print_pval==True:    
                txt = (f'{str(round(param, round_param)) + str(stars)}'
                       f'({str(round(pval, round_param))})')
            elif print_pval==False:
                txt = (f"{str(round(param, round_param)) + str(stars)}")
            else:
                raise ValueError('pval parameter misspecified')
            
            summary_table.loc[variable, model_name] = txt

        # Add the R2
        nobs = self.res.nobs
        r2 = f'{round(100*self.res.rsquared,1)} %'
        r2_adj = f'{round(100*self.res.rsquared_adj,1)} %'
        summary_table.loc['R2', model_name] = r2
        summary_table.loc['R2 adjusted', model_name] = r2_adj

        # Add extra information               
        summary_table.loc['Number of observations', model_name] = nobs
                       
        # Customization of variables names, if needed
        rename_d = {'Const':'Intercept',
                    'alpha[1]':'Alpha',
                    'beta[1]':'Beta',
                    'gamma[1]':'Gamma',
                    'omega':'Omega', 
                    'lam': 'Lambda',
                    'nu': 'Nu',
                    'eta':'Eta', 
                    'lambda':'Lambda'}

        if var_d:
            rename_d.update(var_d)

        summary_table = summary_table.rename(rename_d, axis='index')
        summary_table_nna = summary_table.fillna('').copy()

        return(summary_table_nna)
    
    def __plot(self):
        return DistGARCHFitPlot(self)


       
###############################################################################
#%% Class: DistGARCHForecast
###############################################################################
class DistGARCHForecast(object): # Forecast class for the DistGARCHFit class
    """ 
    Forecast from a DistGarchFit model

    General documentation on VaR forecasting
    https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html#Value-at-Risk-Forecasting
    
    Inputs
    ------
    start_date: str
        Starting date for the forecasts. Example: '2020-03-31'

    horizon: int
        Forecasting horizon
        https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html#Fixed-Window-Forecasting

    fmethod: str
        Simulation methods, among 'analytic', 'simulation', 'bootstrap'
        Not all methods are available for all horizons.
        https://arch.readthedocs.io/en/latest/univariate/forecasting.html#simulation-forecasts

    sample_size: int
        Size of the sample for distribution sampling

    """
    # Initialization
    def __init__(self, DistGARCHFit, start_date, horizon, fmethod,
                 sample_size) -> None:

        self.__dict__.update(DistGARCHFit.__dict__) # Import all attributes 
        
        self.start_date = start_date
        self.horizon = horizon
        self.fmethod = fmethod
        self.sample_size = sample_size

        # Construct the forecasted matrix of exogeneous regressors
        self.exog_fcast_d = dict() # Construct the dictionary of regressors

                
        if self.exog_l == None: # No exogeneous regressors
            self.exog_fcast_d = None
            
        elif len(self.exog_l)>0:
            for var in self.exog_l:
                self.exog_fcast_d[var] = self.df_exog.loc[:, [var]].values
        else:
            raise ValueError('List of exogeneous regressors misspecified')
                
        # Run the forecasts from the arch package
        self.forecasts = self.res.forecast(horizon=self.horizon,
                                           x=self.exog_fcast_d, 
                                           start=self.start_date, 
                                           method=self.fmethod,
                                           reindex=True)
                                        
        # Extract the forecasted conditional mean and variance
        self.cond_mean = self.forecasts.mean.loc[start_date:, 'h.1']
        self.cond_var = self.forecasts.variance.loc[start_date:, 'h.1']
        self.cond_vol = np.sqrt(self.cond_var)
        
        avl_idx_l = np.intersect1d(self.df.index, self.cond_mean.index) 
        true_val = self.df.loc[avl_idx_l, self.depvar]
        residuals = (true_val - self.cond_mean)
        norm_true_val = residuals/self.cond_vol
        
        self.dfor = pd.concat([self.cond_mean, self.cond_var, self.cond_vol,
                               true_val, residuals, norm_true_val], 
                              axis=1, keys=['cond_mean', 'cond_var', 'cond_vol',
                                            'true_val', 'residuals', 'norm_true_val'])        

        # Because the distribution are on the normalized values
        self.cond_ppf = lambda q: self.mod.distribution.ppf(q, self.dist_params)
        self.cond_cdf = lambda v: self.mod.distribution.cdf(v, self.dist_params)
        self.cond_pdf = lambda x: np.exp(self.mod.distribution.loglikelihood(
                                    parameters= self.dist_params,
                                    resids=x,
                                    sigma2=1, 
                                    individual=True))
        
        self.y_ppf = lambda q, mu, sigma : mu + sigma*self.cond_ppf(q)
        self.y_cdf = lambda x, mu, sigma : self.cond_cdf((x-mu)/sigma)
        self.y_pdf = lambda x, mu, sigma : np.exp(self.mod.distribution.loglikelihood(
                                    parameters= self.dist_params,
                                    resids=x-mu,
                                    sigma2=sigma**2, 
                                    individual=True))        
        # Plots
        self.plot = self.__plot()
        
        return None
   
    def rmse(self):
        """
        RMSE: Root Main Squared Errors Out of sample prediction of target variable
        """
        resid_df = self.dfor['residuals'].dropna()
        return np.sqrt((resid_df**2).mean())

    # Function to calculate MAE
    def mae(self):
        """
        MAE: Main Absolute Errors Out of sample prediction of target variable
        """
        resid_df = self.dfor['residuals'].dropna()
        return abs(resid_df).mean()
    
    def mape(self):
        """
        MAPE: Mean Absolute Percentage Error Out of sample prediction of target variable
        """
        temp = self.dfor[['true_val', 'cond_mean']].dropna()
        actual = temp.true_val
        predicted = temp.cond_mean
        mask = actual != 0        
        return np.mean(np.abs((actual - predicted)[mask] / actual[mask])) * 100

    def KS_normalized_innovations(self, 
                                  threshold = 0.05,
                                  verbose=True
                                 ) -> tuple:
        """
        Kolmogorov Smirnov to test that the normalized innovations follows the fitted conditional distribution
        .. math::  
            e_t = (y_t- \hat{mu}_t)/\hat{sigma}_t 
        
        Inputs
        ------            
            threshold: float
                test trheshold. Example: 0.05

            verbose: bool
                Display the test result
                
        Outputs
        ------
            tuple(is_identical, pvalue, statistic)
        """       
        
        return self.KS_test(
            self.dfor['norm_true_val'], 
            self.cond_cdf, 
             threshold = threshold, 
             verbose = verbose)

    
    def KS_test(self,
                data,
                distribution,
                threshold = 0.05,
                verbose=True
               ) -> tuple:
        """ 
        Kolmogorov Smirnov Test         
             The null hypothesis is that the two distributions are identical
             Reject the null hypothesis if the pvalue < threshold

        Inputs
        ------
        data: pd.Series | array
            An array of sample data. 
            
        distribution: 
            str: For the name of the distribution like uniform
            callable: the cdf function of the distribution
            
        threshold: float
            test trheshold. Example: 0.05

        verbose: bool
            Display the test result
            
        Outputs
        ------
        tuple (is_identical, pvalue, statistic)

        """
        
        ks_stats = stats.kstest(data, distribution)
        reject = (ks_stats.pvalue < threshold)

        if verbose:
            PassFail = 'Fail' if reject else 'Pass'
            print(f'KS test {PassFail} !!!')           

        return not reject, ks_stats.pvalue, ks_stats.statistic
        
    def generate_sample_target_variable(self):
        """
        Generate a sample of the target variable
            
        Method
        ------
            1. Generate sample_size realization of the fitted conditional distribution of normalized innovations
            2. For every day, get the conditional mean and conditional variance 
            3. Every day, compute a sample of the target variable
            
        .. math::   
            \forall t, \quad  \hat{y}_{t,i}= \hat{\mu_t} + \hat{\sigma_t} + e_i, \quad {i \in [0,sample_size]}
        """
        self.mod.distribution._random_state = self.rs
        err_sampler = self.mod.distribution.simulate(self.dist_params)
        
        # Sampling the error terms and deriving the values for y in a dataframe
        err_sample = err_sampler(self.sample_size)
        # Rescale to get a sample for the true time series
        cond_vol_reshape = self.cond_vol.values.reshape((-1, 1))  # shape (5675, 1)
        err_sample_reshape = err_sample.reshape((1, -1))  # shape (1, 1000)
        cond_mean_reshape = self.cond_mean.values.reshape((-1, 1))

        sample = pd.DataFrame(cond_mean_reshape + np.dot(cond_vol_reshape, err_sample_reshape),
                                   columns=range(self.sample_size),
                                   index=self.dfor.index)
        self.sample = sample        
        return sample
    
    def generate_y_cdf_empirically(self) -> dict:
        """
        Generate the forecasted CDF of the target variable
            
            arch.univariate package only gives the conditional distribution of normalized innovations
            This function reconstruct the complete conditional distribution of the target variable
            
        Method
        ------
            1. Generate sample of the target variable
            2. Calculate the empirical cdf from daily samples of the target variable
            
        Output
        ------
            ecdf_target: dict
            
        """
        
        if 'sample' in self.__dict__.keys():
            pass
        else:           
            sample = self.generate_sample_target_variable()
        
        print('Compute the empirical conditional cdf of the forecast')
        ecdf_target = {}
        for s in tqdm(self.sample.index):
            ecdf_target[s] = ECDF(self.sample.loc[s])
        
        return ecdf_target
        
        
    def compute_pit(self) -> pd.Series:
        """
        Compute the Pit = the daily conditional cumulative density function of the target variable applied to tru values
        
        """        
        
        self.dfor['pit'] = np.nan
        for s in self.dfor.index:
            self.dfor.loc[s, 'pit'] = self.y_cdf(self.dfor.loc[s, 'true_val'], 
                                                 self.dfor.loc[s, 'cond_mean'],
                                                 self.dfor.loc[s, 'cond_vol'])
        pits = self.dfor['pit']
        return pits
    
    def KS_PIT_test(self, threshold = 0.05, verbose=True) -> tuple:
        """
        Kolmogorov Smirnov to test if the PIT is well specified
        PIT: The cumulative density function of the daily conditional distribution of true values of the target variable has a uniform distribution.
        
        Inputs
        ------            
            threshold: float
                test trheshold. Example: 0.05

            verbose: bool
                Display the test result
        
        Outputs
        ------
            tuple (is_identical, pvalue, statistic)
        """       
        
        if 'pit' in self.dfor.columns:
            pits  = self.dfor['pit'].copy()
        else:      
            pits = self.compute_pit()        
            
        return self.KS_test(
            pits, 
            'uniform', 
            threshold = threshold, 
            verbose = verbose)
        
    def Rossi_Shekopysan_PIT_test(self,
                                  threshold=0.05,
                                  part_distribution='Tails',
                                  plot=False,
                                  verbose=False
                                 ) -> bool:
        '''
        Rossi and Shekopysan JoE 2019 probability integaral transform test
        
            Rossi and Shekopysan prooves that the probability is correctly specified if and only if
            the cumulative density function of the daily conditional distribution of true values of the target variable has a uniform distribution.
            Rossi and Shekopysan provides confidence interval to test for exceedance

        Input:
        ------
            pits: series or list
                The conditional CDF applied to the target variable 
            
            threshold: float. 
                statistical test threshold. Values available [0.01, 0.05, 0.1]
            
            part_distribution: str.
                Part of the distribution on which the specification test is conducted
                    whole Distribution                [0; 0:25]
                    Left Tail                         [0; 0:25]
                    Left Half                         [0; 0:50]
                    Right Half                        [0:50; 1]
                    Right Tail                        [0:75; 1]
                    Center                           [0:25; 0:75]
                    Tails                        [0; 0:25] + [0:75; 1]
            
            plot: bool.
                If True:
                    Plot the empirical cumulative density function of the daily conditional distribution to true values of the target variable
                    Compare them visually to the CDF of a uniform distribution                        
                    The distribution is well specified if and only if the empirical plot do not exceed the Rossi and Shekopysan confidence interval
                
                Else:
                    Do nothing

        Output:
        -------
            Bool: True if the distribution is well specified. Meaning the CDF(PITs) do not breach the confidence interval
                  Esle False
        '''
        if 'pit' in self.dfor.columns:
            pits  = self.dfor['pit'].copy()
        else:      
            pits = self.compute_pit()        
        
        # Compute the ecdf on the pits
        ecdf = ECDF(pits)
        
        # Data work (note that the pit are computed by default)
        support = np.arange(0, 1, 0.01)
        
        # Fit it on the support
        pit_line = ecdf(support)
        uniform_line = stats.uniform.cdf(support)

        # Confidence intervals based on Rossi and Shekopysan JoE 2019
        confidence_interval = pd.DataFrame({
            'part of distribution': ['whole Distribution', 'Left Tail', 'Left Half', 'Right Half', 'Right Tail', 'Center', 'Tails'],
            'Interval': ['[0; 0:25]', '[0; 0:25]', '[0; 0:50]', '[0:50; 1]', '[0:75; 1]', '[0:25; 0:75]', '[0; 0:25] + [0:75; 1]'],
            0.01: [1.61, 1.24, 1.54, 1.53, 1.24, 1.61, 1.33],
            0.05: [1.34, 1.00, 1.26, 1.25, 1.00, 1.33, 1.10],
            0.10: [1.21, 0.88, 1.12, 1.12, 0.88, 1.19, 0.99]
        }).set_index('part of distribution')

        available_thresholds = [0.01, 0.05, 0.1] 
        available_part_distrib = confidence_interval.index
        assert threshold in available_thresholds, f'Test only available on {available_thresholds}'
        assert part_distribution in available_part_distrib, f"part_distribution should belong to one of the following options {available_part_distrib}"

        kp = confidence_interval.loc[part_distribution, threshold]/np.sqrt(len(pits))          
        breach = any(abs(pit_line - uniform_line) > kp)
        is_well_specified = not breach

        if verbose:
            PassFail='Fail' if breach else 'Pass';
            spec = 'not' if breach else '';

            m1= '#'*10+' Rossi and Shekopysan JoE 2019 PIT test ' +'#'*10
            m2 = f'PIT test {PassFail} !!!'
            m3 = f'Distribution is {spec} well specified'
            m4 = f'The conditional CDF of Normalized True Values are {spec} a Uniform'
            print(m1+'\n'+m2+'\n'+m3+'\n'+m4)        

        if plot:
            ci_u = [x+kp for x in support]
            ci_l = [x-kp for x in support]

            fig, ax = plt.subplots(1, 1)
            ax.plot(support, pit_line, color='blue', label='Empirical CDF', lw=2)
            ax.plot(support, uniform_line, color='red', label='Theoretical CDF')
            ax.plot(support, ci_u, color='red', label=f'{int(threshold*100)}% critical values', linestyle='dashed')
            ax.plot(support, ci_l, color='red', linestyle='dashed')
            ax.legend()
            ax.set_xlabel('Quantiles', labelpad=20)
            ax.set_ylabel('Cumulative probability', labelpad=20)
            ax.set_title(('Out-of-sample conditional density:\n'
                          'Probability Integral Transform (PIT) test'), y=1.02)
            plt.show()

        return is_well_specified                     
        

    def log_score(self, aggr_func='EW'):
        """
        Model Performance:
            log score on the entire distribution
            
        Input:
        -----
            aggr_func:str
                the function to aggregate the individual logscores
                EW: Average on the histrory 
                EWMA: Exponentiall Weighted Moving Average on the history        
        """
        
        log_score = self.mod.distribution.loglikelihood(
            parameters=self.dist_params,
            resids = self.dfor['norm_true_val'],
            sigma2= self.cond_var, 
            individual = True
        ).sort_index()
        
        if aggr_func == 'EWMA':
            return log_score.ewm(halflife=int(len(log_score)/4)).mean().iloc[-1]
        
        elif aggr_func == 'EW':
            return log_score.mean()
        
        else:
            raise ValueError("aggr_func not implemented. Supported values are ['EW', EWMA]")
    
    def tailed_log_score(self, area:list =[0, 0.25], area2:list =None, aggr_func = 'EW'):
        
        assert len(area)==2, "area object is a list of 2 elements [start, end]"
        assert area[0] < area[1], "area = [start, end]. where start < end"
        
        if area2 != None:
            assert len(area2)==2, "area object is a list of 2 elements [start, end]"
            assert area2[0] < area2[1], "area = [start, end]. where start < end"
            y_area = pd.concat([
                self.cond_vol*self.cond_ppf(area[0]), 
                self.cond_vol*self.cond_ppf(area[1]),
                self.cond_vol*self.cond_ppf(area2[0]), 
                self.cond_vol*self.cond_ppf(area2[1]),
                self.dfor.true_val
            ], axis=1, keys=['y_area_start', 'y_area_end', 'y_area2_start', 'y_area2_end', 'y_true'])
            cond1 = (y_area['y_area_start'] <= y_area['y_true']) & (y_area['y_true'] <= y_area['y_area_end' ])
            cond2 = (y_area['y_area2_start'] <= y_area['y_true']) & (y_area['y_true'] <= y_area['y_area2_end' ])
            y_area['IsInArea'] = (cond1 | cond2).astype('int')
            
            log_area_integral = np.log(area[1] - area[0]) + np.log(area2[1] - area2[0])
        else:
            y_area = pd.concat([self.cond_vol*self.cond_ppf(area[0]), 
                                self.cond_vol*self.cond_ppf(area[1]),
                                self.dfor.true_val
                               ], 
                               axis=1, 
                               keys=['y_area_start', 'y_area_end', 'y_true']
                              )
            cond1 = (y_area['y_area_start'] <= y_area['y_true']) & (y_area['y_true'] <= y_area['y_area_end' ])
            y_area['IsInArea'] = cond1.astype('int')
            log_area_integral = np.log(area[1] - area[0])
        
        log_score = self.mod.distribution.loglikelihood(
            parameters=self.dist_params,
            resids = self.dfor['residuals'],
            sigma2= self.cond_var, 
            individual = True
        ).sort_index()
        
        tailed_logscore = (log_score - log_area_integral) * y_area['IsInArea']
        
        if aggr_func == 'EWMA':
            return tailed_logscore.ewm(halflife=int(len(tailed_logscore)/4)).mean().iloc[-1]
        
        elif aggr_func == 'EW':
            return tailed_logscore.mean()
        
        else:
            raise ValueError("aggr_func not implemented. Supported values are ['EW', EWMA]")
        
    def fixed_thresholds_FXI(self, thresholds_t):
        """ 
        Return a dataframe with fixed thresholds 

        Inputs
        ------

        thresholds_t: tuple
            Tuple of intervention threshold, one negative, one positive
                  
        """

        assert isinstance(thresholds_t, tuple), 'Thresholds should be tuple'
        assert thresholds_t[0] < 0, 'First threshold should be negative'
        assert thresholds_t[1] > 0, 'Second threshold should be negative'
        
        # Prepare the frame
        dv = pd.merge(self.dfor, self.df[[self.depvar, self.level]],
                      left_index=True, right_index=True)

        # Determine which values are below or above thresholds

        c_below = (dv['FX log returns'] <= thresholds_t[0])
        c_above = (dv['FX log returns'] >= thresholds_t[1])

        # FX interventions
        dv['FXI'] = 'No'                
        dv.loc[c_below, 'FXI'] = 'Below'
        dv.loc[c_above, 'FXI'] = 'Above'

        # Logreturns where the FXI occur
        dv[f'Logret Below'] = np.nan
        dv.loc[c_below, f'Logret Below'] = dv.loc[c_below, 'FX log returns']
        dv[f'Logret Above'] = np.nan
        dv.loc[c_above, f'Logret Above'] = dv.loc[c_above, 'FX log returns']  
        
        # Level where the FXI occur
        dv[f'Level Below'] = np.nan
        dv.loc[c_below, f'Level Below'] = dv.loc[c_below, self.level]
        dv[f'Level Above'] = np.nan
        dv.loc[c_above, f'Level Above'] = dv.loc[c_above, self.level]
    
        return(dv)


    def VaR_FXI(self, qv_l=[0.025, 0.975]):
        """ 
        Return a dataframe with VaR thresholds

        Inputs
        ------

        qv_l: List
            List of quantiles
                  
        """

        qv_l = sorted(qv_l)
        qv_labels_l = [f'cond_quant_{x:g}' for x in qv_l]
        
        # Prepare the frame
        dv = pd.merge(self.dfor, self.df[[self.depvar, self.level]],
                      left_index=True, right_index=True)

        # Single flag
        dv['FXI'] = np.nan
        
        # Create the exceedance markers, based on quantiles list
        dv['Below'] = np.nan
        c_below = (dv[self.depvar] <= dv[f'{qv_labels_l[0]}'])
        dv.loc[c_below, 'Below'] = dv.loc[c_below, self.depvar]
        dv.loc[c_below, 'FXI'] = 'Below'

        dv['Above'] = np.nan
        c_above = (dv[self.depvar] >= dv[f'{qv_labels_l[1]}'])
        dv.loc[c_above, 'Above'] = dv.loc[c_above, self.depvar]
        dv.loc[c_above, 'FXI'] = 'Above'
                        
        if self.level: # If level, add it
            dv[f'Level Below'] = np.nan
            dv.loc[c_below, f'Level Below'] = dv.loc[c_below, self.level]

            dv[f'Level Above'] = np.nan
            dv.loc[c_above, f'Level Above'] = dv.loc[c_above, self.level]

        return(dv)
    
    def __plot(self):
        return DistGARCHForecastPlot(self)

    
###############################################################################
#%% Class: DistGARCHOptimize
###############################################################################    
class DistGARCHOptimize:
    """
    Optimize the GARCH model for this time series
    
    """
    __description = "Optimize the fit and forecasts"
    __author = "Romain Lafarguette - romainlafarguette@github.io, Amine Raboun - amineraboun@github.io"
    
    def __init__(self, DistGARCH):
        self.__dict__.update(DistGARCH.__dict__)
        
    def extract_mean(self, exog_l:list, lags_l:list, fv=ConstantVariance(), verbose=False):
        """
        Test Mean Model given a list of exogenous variable and lags
        
        Input:
        ------
            exog_l: list
                list of exogenous variables
            
            lags_l: list
                list of integer values giving the lags
                
            fv: FixedVariance Object
                The ConstantVariance is special case

        Output:
        -------
            performance: pd.Series
                Performance metrics of the tested model:
                    In-Sample = R2, R2_adj, AIC, BIC
                    Out-of-Sample = RMSE, MAE, MAPE
            
            mean_mod: DistGARCH
                model
                
            mean_fit: DistGARCHFit
                result of the fit of the model
                
            mean_forecast: DistGARCHForecast
                result of the forecst of the model
        """
        
        # Mean (drift) model, potentially with exogeneous regressions and lags
        mean_mod = DistGARCH(depvar_str=self.depvar,
                                  data=self.df,
                                  level_str=self.level,                
                                  exog_l= exog_l, # Defined above 
                                  lags_l= lags_l, 
                                  vol_model= fv,
                                  dist_family=Normal()
                                 )
        mean_fit = mean_mod.fit(disp='off', verbose=verbose)
        if isinstance(lags_l, list):
            n= max(lags_l)
        else:
            n = lags_l
            
        mean_forecast = mean_fit.forecast(start_date= self.df.index[n])
        performance = pd.Series({
                 'R2': mean_fit.res.rsquared, 
                'R2_adj': mean_fit.res.rsquared_adj,
                'AIC': mean_fit.res.aic, 
                'BIC': mean_fit.res.bic,                
                'RMSE': mean_forecast.rmse(), 
                'MAE': mean_forecast.mae(), 
                'MAPE': mean_forecast.mape()
               })
            
        return performance, mean_mod, mean_fit, mean_forecast
    
    def optimize_mean(self, verbose=False):
        """
        Search for the best Out-of-Sample Test Mean Model given a list of exogenous variable and lags

        Output:
        -------
            performance: pd.DataFrame
                Summary Table on the out-of-sample performance of all possible combinations of exogenous variables if provided                    
            
            performance_lags: pd.DataFrame
                Summary Table on lags from 1 to 10
            
        """
        if self.exog_l == None:
            self.best_combination = self.exog_l
            
        else:
            # Take the best combination of exogenous variables
            combinations = []
            for i in range(1, len(self.exog_l)+1):
                combinations += list(itertools.combinations(self.exog_l, i))

            performance = {}
            print('Optimizing the Mean model. step 1: exog_l')
            for exog_l in tqdm(combinations):
                exog_l = list(exog_l)
                performance[','.join(exog_l)], _, _, _ = self.extract_mean(exog_l=exog_l, lags_l=[1], verbose=verbose)

            performance = pd.concat(performance.values(), keys = performance.keys(), axis=1)
            best_combination = performance.T.sort_values('RMSE').index[0]
            print(f'Best Out-Of-Sample combination of exogenous variables: \n{best_combination}')
            self.best_combination = [c.strip() for c in best_combination.split(',')]
        
        # Given the best combination, finetoone the lags
        performance_lags = {}
        print('Optimizing the Mean model. step 2: number of lags')
        for lags_l in tqdm(range(10)):
            performance_lags[lags_l], _, _, _ = self.extract_mean(exog_l=self.best_combination, lags_l=lags_l)
            
        performance_lags = pd.concat(performance_lags.values(), keys = performance_lags.keys(), axis=1)
        # To avoid chosing the model with the highest lags just because we add lags
        # impose to increase the RMSE with at least 1%
        smallest_rmse = (performance_lags.loc['RMSE']/performance_lags.loc['RMSE', 0]).sort_values()
        if smallest_rmse.iloc[0]< (1-0.01):
            best_lag = smallest_rmse.index[0]
        else:
            best_lag = 0
        best_lag = performance_lags.T.sort_values('RMSE').index[0]
        print(f'Best Out-Of-Sample number of lags: {best_lag}')
        self.best_lag = best_lag
        
        self.mean_perf, self.mean_mod, self.mean_fit, self.mean_forecast = self.extract_mean(exog_l= self.best_combination, 
                                                                                             lags_l= self.best_lag)
        self.epsilon = self.mean_fit.res.resid.dropna()        
        return performance, performance_lags
    
    def residuals_moments(self, residuals, verbose=False):
        """
        Compute the moments of the residuals
        """
        
        moments = {'mean': residuals.mean(), 
                   'std':residuals.std(),
                   'variance':residuals.var(),
                   'skew':residuals.skew(), 
                  'kurt':residuals.kurt()}
        if verbose:
            for k, v in moments.items():
                print(f'{k} =  {np.round(v,2)}')
            if (moments['kurt'] > 1) or (moments['skew'].abs()>1):               
                print('As expected residuals have high skewness and fat tails')
                
        return moments

    def assess_vol_dist_model(self, 
                              residuals,
                              volatility_model,
                              distribution_family, 
                              threshold =0.05
                              ):
        """
        Test Volatility and distribution Model 
        
        Input:
        ------
            volatility_model: arch.univariate object
                ConstantVariance, ARCH, EGARCH, GARCH, GARCH, EWMAVariance, RiskMetrics2006 # Volatility
            
            distribution_family: arch.univariate object
                Normal, StudentsT, SkewStudent, GeneralizedError # Distributions
                
            threshold: float
                threshold for the statistics [0.1, 0.05, 0.01]
                
            tails: str
                tails on which the focus of the test is conducted [Tails, Left Tail, Right Tail]

        Output:
        -------
            performance: pd.Series
                Performance metrics of the quality of the distribution fit:
                    Model Specification:
                        KS_normalized_innovations: Kolmorovo Smirnov to test if the In-sample normalized innovation follows the fitted distribution
                        KS_PIT_test: Kolmorovo Smirnov to test if the cdf of the predicted distribution applied to true values is a uniform
                        Rossi_Shekopysan_PIT_test: Test if the cdf of the predicted distribution applied to true values breaches the RS confidence interval
                    Model Evaluation:
                        log_score: the log likelihood of the true values have been generated by the specified distribion
                        tailed_log_score: same as log scores but focused on the tails alone, where we are interested to be accurate in order to compute the VaR
            
            vol_mod: DistGARCH
                model
                
            vol_fit: DistGARCHFit
                result of the fit of the model
                
            vol_forecast: DistGARCHForecast
                result of the forecst of the model
        """
                
        vol_mod = DistGARCH(depvar_str='residuals',
                            data=residuals,
                            vol_model= volatility_model,
                            dist_family= distribution_family, 
                            init_type = 'ZEROMEAN'
                           )
        vol_fit = vol_mod.fit(verbose=False)

        vol_forecast = vol_fit.forecast(start_date= self.epsilon.dropna().index[0])
            
        KS_normalized_innovations, KS_normalized_innovations_pvalues, _  = vol_forecast.KS_normalized_innovations(threshold=threshold, verbose=False)
        KS_PIT_test, KS_PIT_test_pvalues, _ = vol_forecast.KS_PIT_test(threshold=threshold, verbose=False)
        log_score = vol_forecast.log_score(aggr_func='EWMA')    
        
        performance = {
            'KS_normalized_innovations': KS_normalized_innovations,
            'KS_normalized_innovations_pvalues': KS_normalized_innovations_pvalues,
            'KS_PIT_test': KS_PIT_test,
            'KS_PIT_test_pvalues': KS_PIT_test_pvalues,
            'log_score': log_score
                  }
        
        # Distribution Tails Specification and performance
        def _get_areas_per_tail(tails):
            # Model Performance
            if tails =='Tails':
                area =[0, 0.25];
                area2=[0.75, 1]
            elif tails == 'Left Tail':
                area =[0, 0.25];
                area2=None
            elif tails == 'Right Tail':
                area =[0.75, 1];
                area2=None
            else:
                recognized_tails = ['Tails', 'Left Tail', 'Right Tail']
                raise ValueError(f'Warning !! tails not recognized {recognized_tails}')
            return area, area2
                
        tailed_log_score = {}
        RS_test_PIT = {}
        for tail in ['Tails', 'Left Tail', 'Right Tail']:
            tailname = '_'.join(tail.lower().split(' '))
            RS_test_PIT[f'Rossi_Shekopysan_PIT_test_{tailname}'] = vol_forecast.Rossi_Shekopysan_PIT_test(part_distribution = tail, 
                                                             threshold = threshold,
                                                             verbose=False, plot=False)
            area, area2 = _get_areas_per_tail(tail)
            tailed_log_score[tailname+'_log_score'] = vol_forecast.tailed_log_score(area=area, area2=area2, aggr_func='EWMA')            
            
            
        
        performance = {**performance, **RS_test_PIT, **tailed_log_score}
        return performance, vol_mod, vol_fit, vol_forecast
        
    def optimize_vol_distrib(self,
                             threshold =0.05, 
                             tails='Tails', 
                             optimize_mean=True): 
        """
        Search for the best Out-of-Sample Volatility and Distribution
            Test 2-by-2 combinations of 
                Volatility Models: 'Constant', 'ARCH', 'EGARCH', 'GARCH', 'GJR-GARCH', 'EWMA', 'RiskMetric'
                Distribution Families: 'Normal', 'StudentT', 'SkewStudent', 'GeneralizedError'
                
            Choose among the well specified model, i.e, those passing the pis and kolmogorov test, 
                the model with the best performance on the complete distribution based on the log_score and on the tails based on the tailed_log_score
                if no combination is well specified, return the combination with best performance regardless if specified or not

        Output:
        -------
            performance: pd.DataFrame
                Summary Table on the out-of-sample performance of all possible combinations of GARCH type volatility models and Distributions 
            
        """
        
        if 'epsilon' in self.__dict__.keys():
            pass
        
        else:
            if optimize_mean:
                self.optimize_mean()
            else:
                _, _, mean_fit, _ = self.extract_mean(exog_l=self.exog_l, lags_l=[1]) 
                self.epsilon = mean_fit.res.resid.dropna()                        
            
        vol_spec_l = [ConstantVariance(), ARCH(1), EGARCH(1,1,1), GARCH(1), GARCH(1,1), EWMAVariance(None), RiskMetrics2006()]
        vol_labels_l = ['Constant', 'ARCH', 'EGARCH', 'GARCH', 'GJR-GARCH', 'EWMA', 'RiskMetric']
        vol_mod_dict = {k:v for k, v in zip(vol_labels_l, vol_spec_l)}

        # List of error distribution
        errdist_l = [Normal(), StudentsT(), SkewStudent(), GeneralizedError()]
        errdist_labels_l = ['Normal', 'StudentT', 'SkewStudent', 'GeneralizedError']
        distribution_dict = {k:v for k, v in zip(errdist_labels_l, errdist_l)}

        performance = {}
        with tqdm(total=len(vol_mod_dict)*len(distribution_dict)) as pbar:
            for vol_name, volatility_model in vol_mod_dict.items():
                for distribution_name, distribution_family in distribution_dict.items(): 
                    tmpperf, _, _, _ = self.assess_vol_dist_model(self.epsilon, volatility_model, distribution_family)                
                    tmpperf['volatility_model'] = vol_name
                    tmpperf['distribution'] = distribution_name
                    performance[f'{vol_name} - {distribution_name}'] = pd.Series(tmpperf)
                    pbar.update(1)

        performance = pd.concat(performance.values(), axis=1).T
        
        tailname = '_'.join(tails.lower().split(' '))
        specification_on_tails = f'Rossi_Shekopysan_PIT_test_{tailname}'
        rank_on_tails = f'{tailname}_log_score'
        def get_best(_df):            
            assert rank_on_tails in _df.columns, f'{rank_on_tails} must be in the performance summary'
            assert 'log_score' in _df.columns, 'log_score must be in the performance summary'            
            
            _df = _df.sort_values([rank_on_tails, 'log_score'], ascending=[False, False])            
            self.best_vol_mod_name = _df['volatility_model'].iloc[0]
            self.best_vol_mod = vol_mod_dict[self.best_vol_mod_name]
            
            self.best_distrib_name = _df['distribution'].iloc[0]
            self.best_distrib = distribution_dict[self.best_distrib_name]
            
            print(f'Best Out-Of-Sample Volatility Model: {self.best_vol_mod_name}')        
            print(f'Best Out-Of-Sample Distribution Family: {self.best_distrib_name}')    
            
        cond1 = performance[specification_on_tails]
        cond2 = performance['KS_PIT_test']
        cond3 = performance['KS_normalized_innovations']    
        
        well_specified = performance[cond1 & cond2 & cond3]
        if well_specified.empty:
            print('No combination of Volatility Model and Distribution Family capture the heteroscedasticity of the residuals')
            if performance[cond1].empty:
                print('take the best log score regardless of the specification')
                get_best(performance)                
            
            else:
                get_best(performance[cond1])
        else:
            get_best(well_specified)           
            
        column_display = [
            'KS_normalized_innovations', 'KS_normalized_innovations_pvalues',
             'KS_PIT_test', 'KS_PIT_test_pvalues', 
             'Rossi_Shekopysan_PIT_test_tails', 'Rossi_Shekopysan_PIT_test_left_tail', 'Rossi_Shekopysan_PIT_test_right_tail', 
             'log_score', 'tails_log_score', 'left_tail_log_score', 'right_tail_log_score']
        sort_order = [specification_on_tails,  'KS_PIT_test',  'KS_normalized_innovations', 'log_score', rank_on_tails]
        return performance.set_index(['volatility_model', 'distribution'])[column_display].sort_values(sort_order)
    
    def fine_tune_model(self, max_iter:int=10, convergence_rate:float=0.01)->pd.DataFrame:
        """
        Apply The Zig-Zag Method on the best model for mean and variance to stabilize parmeters
        
        Input:
        ------
            max_iter: int
                Maximum Number of iterations before breaking from the loop
                
            convergence_rate: float, example 0.01 for 1%
                The maximum percentage differnence between two consecutive update of parameters                
        
        Output:
        -------
            param_df: pd.DataFrame
                Fitted parameters at each iteration
        """
        if 'best_combination' in self.__dict__:
            pass
        else:
            self.optimize_mean()
            
        if 'best_vol_mod' in self.__dict__:
            pass
        else:
            self.optimize_vol_distrib()
           
        
        print('Fine Tune the best model')
        print(f'exog_l = {self.best_combination}')
        print(f'lags = {self.best_lag}')
        print(f'volatility model = {self.best_vol_mod_name}')
        print(f'distribution family = {self.best_distrib_name}') 
        
        print('Stabilize Parameters with ZigZag method ...')
        iter_params = {}; converged=False
        for i in range(max_iter):
            if i==0:
                fv = ConstantVariance()
            else:
                # Create the fixed variance
                cond_var = vol_fit.res.conditional_volatility**2
                variance = pd.Series(index=self.df.index, dtype='float64')                
                variance.loc[cond_var.index] = cond_var
                variance = variance.bfill()
                fv = FixedVariance(variance, unit_scale=True)

            # Re-train the main model with the previous Conditional Variance
            performance, mean_mod, mean_fit, mean_forecast = self.extract_mean(exog_l= self.best_combination,
                                                                               lags_l=self.best_lag, 
                                                                               fv= fv)
            performance, vol_mod, vol_fit, vol_forecast = self.assess_vol_dist_model(residuals = mean_fit.res.resid.dropna(),
                                                                                     volatility_model= self.best_vol_mod,
                                                                                     distribution_family = self.best_distrib)

            mean_params = mean_fit.res.params.drop('sigma2') if (i==0) else mean_fit.res.params
            mean_pvalues = mean_fit.res.pvalues.drop('sigma2') if (i==0) else mean_fit.res.pvalues            
            
            iter_params[f"iteration_{i}"] = pd.concat([
                pd.concat([mean_params, vol_fit.res.params]),
                pd.concat([mean_pvalues, vol_fit.res.pvalues])
            ], axis=1, keys=['params', 'pvalues'])
            
            if i>=1:
                prev_param = iter_params[f"iteration_{i-1}"].params
                new_params = iter_params[f"iteration_{i}"].params
                pct_param_shift = ((new_params-prev_param)/prev_param).abs().max()
                if pct_param_shift < convergence_rate:
                    converge=True
                    break;
                else:
                    continue
        if converge:
            print('Converged !')
        else:
            print('Max iteration reached without convergion !')

        params_df = pd.concat(iter_params.values(), keys = iter_params.keys(), axis=1)
        
        self.final_model = DistGARCH(
            depvar_str=self.depvar,
              data=self.df,
              level_str=self.level,                
              exog_l= self.best_combination, # Defined above 
              lags_l= self.best_lag, 
              vol_model= self.best_vol_mod,
              dist_family=self.best_distrib,
             init_type='FIXED',
            params = iter_params[f"iteration_{i}"]['params']
            )
        return params_df      
            
###############################################################################
#%% Descriptive Plots
###############################################################################        
class DistGARCHPlot:
    """
    Present a series of descriptive statitics upon initializing the DistGarch
    
    """
    __description = "Plot the descriptive statistics"
    __author = "Romain Lafarguette - romainlafarguette@github.io, Amine Raboun - amineraboun@github.io"
    
    def __init__(self, DistGARCH):
        self.__dict__.update(DistGARCH.__dict__)
        
    # Public methods
    def plot_description(self, start_date=None,
                         title_returns = 'Historical returns',
                         title_level = 'Historical level',
                         title_density = 'Historical distribution of returns',
                         y_label_returns = 'bps',
                         y_label_level = 'FX rate',
                         xticks_freq=None):
        """ 
        Descriptive plot: returns and level, with distribution

        start_date: str, default None
            To restrict the sample after a given date. Enter as "2015-03-31"
        """

        # Restrict the sample if needed
        if start_date:
            data = self.df.loc[start_date:, :].copy()
        else:
            data = self.df.copy()
            
        # Prepare the plots
        if self.level:
            fig, (ax1, ax2, ax3) = plt.subplots(3,1)

        else:
            fig, (ax2, ax3) = plt.subplots(2,1)

        # First plot: level
        if self.level:
            ax1.plot(data.index, data[self.level])
            ax1.set_title(title_level, y=1.02)
            ax1.set_xlabel('')
            ax1.set_ylabel(y_label_level, labelpad=20)

        # Second plot: Returns
        ax2.plot(data.index, data[self.depvar])
        ax2.set_title(title_returns, y=1.02)
        ax2.set_xlabel('')
        ax2.set_ylabel(y_label_returns, labelpad=20)

        # Manage frequency of xticks & make sure the last one always visible
        if xticks_freq:
            start, end = ax1.get_xlim()
            t_seq = np.append(np.arange(start, end-5, xticks_freq), end)
            ax1.xaxis.set_ticks(t_seq)
            ax2.xaxis.set_ticks(t_seq)

        # Third plot: Returns Density
        sns.histplot(data[self.depvar], bins=100, ax = ax3)
        ax3.set_title(title_density, y=1.02)
        ax3.set_xlabel(y_label_returns)
        
        # Adjust
        plt.subplots_adjust(hspace=0.5)

        # Exit
        return(fig)
###############################################################################
#%% Fit Plots
###############################################################################       
class DistGARCHFitPlot:
    """
    Present a series of plots describing the googdeness of fit
    """
    
    __description = "Plot the descriptive statistics"
    __author = "Romain Lafarguette - romainlafarguette@github.io, Amine Raboun - amineraboun@github.io"
    
    def __init__(self, DistGARCHFit):
        self.__dict__.update(DistGARCHFit.__dict__)
        
    # Public methods: Plots
    def plot_summary_fit(self):
        """ Summary fit plot: residuals and conditional volatility """
        # Prepare the plot
        fig = plt.figure()        
        self.res.plot()
        return(fig)
    
    # In-sample conditional volatility
    def plot_in_cond_vol(self,
                         start_date=None, 
                         title='In sample conditional volatility',
                         ylabel='Conditional volatility',
                         xticks_freq=None):
        
        """ 
        Plot the (estimated) in-sample conditional volatility 

        start_date: str, default None
            Start date to restrict the sample, str should be '2015-03-31'

        xticks_freq: int or None, default None
            E.g. take one ticks over 5 (None to use plt default)

        """

        # Conditional volatility data
        cv = self.res.conditional_volatility.dropna().copy()

        if start_date:
            cv = cv.loc[start_date:].copy()
                        
        fig, ax1 = plt.subplots(1,1)

        ax1.plot(cv.index, cv, lw=3)
        ax1.set_title(title, y=1.02)
        ax1.set_xlabel('')
        ax1.set_ylabel(ylabel, labelpad=20)
        
        # Manage frequency of xticks & make sure the last one always visible
        if xticks_freq:
            start, end = ax1.get_xlim()
            t_seq = np.append(np.arange(start, end-5, xticks_freq), end)
            ax1.xaxis.set_ticks(t_seq)

        ax1.tick_params(axis='x', rotation=45)    
        return(fig)
    
           
    # Plot the relationship between past innovations and volatility
    def plot_shocks_vol(self,
                        vol_thresholds_l=[1.5, 3],
                        xlabel='Shocks (t-1)',
                        ylabel='Normalized conditional volatility (t, std)', 
                        title='Shocks & conditional volatility',
                        subtitle='With polynomial fit & intervention thresholds',
                        x_range=None, 
                        range_share=0.05):

        """ 
        Scatter Plot of Conditional Volatility & Innovations 
        With polynomial and tangent

        vol_thresholds_l: list of float
            Threshold for volatility. Should be 1.5 for 50% more than min one
        
        share_range: float, between 0 and 1, default 0.1
            Length of the tangent lines, expressed as share of x-data range

        """
                      
        #### Data work
        ds = pd.merge(self.res.resid, self.in_cond_vol,
                      left_index=True, right_index=True)
        ds.columns = ['shocks', 'cond_vol']

        ds = ds.dropna().copy()        
        
        # Normalize volatility by a constant to improve the charts
        # Look at the volatility with the constant of a quadratic fit
        qparams = np.polyfit(ds['shocks'], ds['cond_vol'], 2)
        constant = qparams[2] # "c" in : ax2 + bx + c

        # Look at the volatility in normal times, 
        #normalization = np.percentile(ds['cond_vol'], 50)
        ds['norm_cond_vol'] = ds['cond_vol']/constant

        # Data
        x_data = ds['shocks'].values
        y_data = ds['norm_cond_vol'].values

        # Polynomial fit
        params = np.polyfit(x_data, y_data, 2)

        if x_range:
            x_support = np.linspace(x_range[0], x_range[1], 1000)
        else:
            x_support = np.linspace(min(x_data), max(x_data), 1000)
            
        y_fit = np.polyval(params, x_support)
           
        #### Plots        
        # Prepare the plot
        fig, ax = plt.subplots(1,1)

        # Draw original data as a scatter plot
        ax.scatter(x_data, y_data, color='blue')
        ax.plot(x_support, y_fit, color='green', lw=3)

        t_idx_l = list() # Container to store threshold lists
        for thresh in vol_thresholds_l:
            ax.axhline(y=thresh, linestyle='--', color='darkred')
          
            # Point where the intersection occurs
            inter_idx = np.argwhere(np.diff(np.sign(thresh - y_fit))).flatten()
            for idx in inter_idx:
                t_idx_l.append(idx)
                ax.vlines(x=x_support[idx], ymin=0, ymax=y_fit[idx],
                          color='darkred', linestyle='--')
                         
        ax.scatter(x_support[t_idx_l], y_fit[t_idx_l], s=100, c='darkred')  

        # Arrange the y-ticks        
        ax.yaxis.set_ticks(np.arange(0, max(y_fit) + 0.5, 0.5))
        
        # Remove the standard x-ticks
        ax.set_xticks([])
        new_ticks_l = sorted([int(x) for x in x_support[t_idx_l]])
        ax.set_xticks(new_ticks_l + [0]) # Add new ticks

        extra_idx_l = [new_ticks_l.index(x) for x in new_ticks_l]
        for idx in extra_idx_l: 
            ax.get_xticklabels()[idx].set_color("darkred")
            #ax.get_xticklabels()[idx].set_fontproperties('bold')
        
        # Some customization
        ax.set_xlabel(xlabel, labelpad=20) # X axis data label
        ax.set_ylabel(ylabel, labelpad=20) # Y axis data label
        
        x_range = max(x_support) - min(x_support)
        plt.xlim(min(x_support) - range_share*x_range,
                 max(x_support) + range_share*x_range)
 
        plt.ylim(0, max(y_fit))

        
        # Title
        plt.title(f'{title} \n {subtitle}', y=1.02)        

        plt.show()

        return(None)
    
###############################################################################
#%% Forecast Plots
###############################################################################
class DistGARCHForecastPlot(object):
    """ 
    Present a series of plot of the DistGARCHForecast Class Above
    Loaded as a class method

    Paratemers:
    
    """
    __description = "Plots the GARCH forecasted distribution"
    __author = "https://amineraboun.github.io/"

    def __init__(self, forecasted:DistGARCHForecast): 
        self.__dict__.update(forecasted.__dict__) # Import all attributes 
        self.forecasted = forecasted
        
        #### CDF and sampling estimation
        q_l = [0.01] + [round(x,3) for x in np.arange(0.025,1,0.025)] + [0.99]
        self.cond_quant_labels_l = [f"cond_quant_{q:g}" for q in q_l]     
        err_cond_quant = self.cond_ppf(q_l)
        
        # Summarize the conditional quantiles in a dataframe
        # xbar = (x-mu)/std => x = mu + xbar*std
        # mean + sqrt(cond_var)*the quantiles of the errors terms
        for var in self.cond_quant_labels_l: self.dfor[var] = np.nan
        self.dfor[self.cond_quant_labels_l] = (self.dfor[['cond_mean']].values
                                            + (self.dfor[['cond_vol']].values
                                                  * err_cond_quant[None, :]))
        return None
    
    # Public plot methods
    # Forecasted conditional mean and volatility
    def plot_out_cond_mean_vol(self, ylabel='',
                               title_mean='Out-of-sample conditional mean',
                               title_vol='Out-of-sample conditional volatility', 
                               ylabel_mean='',
                               ylabel_vol='',
                               xticks_freq=None, 
                               hspace=0.3):
        """ 
        Plot the conditional mean and volatility over forecasted periods
        """

        # Prepare the plot and fix the parameters                
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # Conditional mean
        ax1.plot(self.dfor.index, self.dfor['cond_mean'])
        ax1.set_title(title_mean, y=1.02)
        ax1.set_xlabel('')
        ax1.set_ylabel(ylabel_mean, labelpad=20)

        if xticks_freq:
            start, end = ax1.get_xlim()
            t_seq = np.append(np.arange(start, end-5, xticks_freq), end)
            ax1.xaxis.set_ticks(t_seq)

        # Conditional volatility
        ax2.plot(self.dfor.index, self.dfor['cond_vol'])
        ax2.set_title(title_vol, y=1.02)
        ax2.set_xlabel('')
        ax2.set_ylabel(ylabel_vol, labelpad=20)

        if xticks_freq:
            start, end = ax2.get_xlim()
            t_seq = np.append(np.arange(start, end-5, xticks_freq), end)
            ax2.xaxis.set_ticks(t_seq)
        
        # Adjust the space between plots
        plt.subplots_adjust(hspace=hspace)
        
        return(fig)

    # Conditional Quantiles
    def plot_out_condquant(self, quantile=0.05,
                           ylabel='',
                           title='Out-of-sample conditional VaR at 5%'):
        """ 
        Plot the conditional value at risk (quantile) over forecasted periods

        clip: tuple or None, default None
            Tuple to limit the plots

        """
        # Prepare the plot        
        fig = plt.figure()        
        self.dfor[f'cond_quant_{quantile:g}'].plot(lw=3, legend=None)
        plt.xlabel('')
        plt.ylabel(ylabel, labelpad=20)
        plt.title(title, y=1.01)
        return(fig)

    # Multiple distributions plot (Joyplot)
    def plot_joyplot_out(self,
                    title='Conditional forecasted density out-of-sample',
                    label_drop=3,
                    colormap=cm.autumn_r,
                    xlabel='',
                    xlimits_t=None):
        """ 
        Plot a forecasted joyplot (multiple densities plotted over time) 

        label_drop: integer, default 3
            Keep one label every x of them, avoiding crowded chart

        colormap: matplotlib colormap. Default: yellow/red scale
            To have a unique color (e.g. all blue), choose None

        xlimits_t: tuple or None, default None
            Fix the limits for the xscale

        """
        if 'sample' in self.__dict__:
            pass
        else:
            self.sample = self.forecasted.generate_sample_target_variable()
        
        # Arrange the sample frame        
        dsam = self.sample.dropna().transpose().copy()
        dates_l = dsam.columns 
        dsam['id'] = dsam.index
        dlsam = pd.melt(dsam, value_vars=dates_l,
                        id_vars=['id'])
        dlsam = dlsam.rename(columns={'variable':'date'})
        dlsam['fdate'] = dlsam['date'].dt.strftime('%Y-%m-%d') 

        # Colors based on the volatility of each sample (can do on median too)
        vs = pd.DataFrame(dlsam.groupby(['date'])['value'].std())
        vs['date'] = pd.to_datetime(vs.index)
        vs['fdate'] = vs['date'].dt.strftime('%Y-%m-%d') 
        vs = vs.rename(columns= {'value': 'volatility'})

        colorparams = vs['volatility'] # Parameter of the color
        colormap = cm.autumn_r
        normalize = mcolors.Normalize(vmin=np.min(colorparams),
                                      vmax=np.max(colorparams))
        vs['color'] = [mcolors.to_rgba(colormap(normalize(x)))
                       for x in vs['volatility']]

        # Need to map volatility and color
        dall = pd.merge(dlsam, vs, left_on=['fdate'], right_on=['fdate'])
        
        # Arrange the labels, take one every three
        labels_l = list()
        for idx,val in enumerate(sorted(set(dall['fdate']))):
            if idx%label_drop==0:
                labels_l.append(val)
            else:
                labels_l.append(None)

        # Avoid extreme values        
        if isinstance(xlimits_t, tuple):
            dall = dall.loc[dall['value'] >= xlimits_t[0], :].copy()
            dall = dall.loc[dall['value'] <= xlimits_t[1], :].copy()


        # Prepare the plot
        fig, axes = joyplot2(dall, by="fdate", column="value",
                             range_style='own',
                             labels=labels_l, figsize=(12,10),
                             grid="y", linewidth=1, legend=False, 
                             fade=True,
                             color=list(vs['color']))
    
        plt.xlabel(xlabel)
        plt.title(title, y=1.02) # Can not be on left
            
        return(fig)
    
    
    def plot_qqplot_normalized_innovations(self, 
                   xlabel = 'Theoretical Quantiles',
                   ylabel = 'Sample Quantiles',
                   title = 'QQ-Plot of the normalized innovations'
                  ) -> None:
        """
        Plot the QQ plot for the normalized innovations given the estimated distribution.
            A Q-Q plot (quantile-quantile plot) is a graphical method for checking if a sample follows a specified distribution
        

        Inputs
        ------
        xlabel: str
        ylabel: str
        title: str
            Text to put in the plot 

        Returns:
        None
        """
        
        data = self.dfor['norm_true_val'] # An array of sample data.
        ppf = self.cond_ppf
        
        # Calculate percentiles of the sample data
        percentiles = np.linspace(.1, 99, 99)

        # Calculate theoretical quantiles for each percentile of the sample data
        sampling_quantiles = np.array([np.percentile(data, p) for p in percentiles])
        theoretical_quantiles = ppf(percentiles/100)

        # Plot the sample quantiles against the theoretical quantiles
        fig, ax = plt.subplots(1, 1)
        plt.plot(theoretical_quantiles, sampling_quantiles, label= 'Empirical Fit')
        plt.plot(theoretical_quantiles, theoretical_quantiles, color='r', label='Best Fit')
        ax.legend(frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title = title + f'\nAssumed Distribution: {self.mod.distribution.name}'
        ax.set_title(title)
        return (fig)
        
        
    def plot_pit(self,
                  threshold=0.05,
                  part_distribution='Tails',
                  xlabel='Quantiles',
                  ylabel='Cumulative probability',
                  title=('Out-of-sample conditional density:'
                         ' Probability Integral Transform (PIT) test')
                     ):
        '''
        Rossi and Shekopysan JoE 2019 probability integaral transform test
        
            Rossi and Shekopysan prooves that the probability is correctly specified if and only if
            the cumulative density function of the daily conditional distribution of true values of the target variable has a uniform distribution.
            Rossi and Shekopysan provides confidence interval to test for exceedance

        Input:
        ------
            
            threshold: float. 
                statistical test threshold. Values available [0.01, 0.05, 0.1]
            
            part_distribution: str.
                Part of the distribution on which the specification test is conducted
                    whole Distribution                [0; 0:25]
                    Left Tail                         [0; 0:25]
                    Left Half                         [0; 0:50]
                    Right Half                        [0:50; 1]
                    Right Tail                        [0:75; 1]
                    Center                           [0:25; 0:75]
                    Tails                        [0; 0:25] + [0:75; 1]
            
            plot: bool.
                If True:
                    Plot the empirical cumulative density function of the daily conditional distribution to true values of the target variable
                    Compare them visually to the CDF of a uniform distribution                        
                    The distribution is well specified if and only if the empirical plot do not exceed the Rossi and Shekopysan confidence interval
                
                Else:
                    Do nothing

        '''
        
        if 'pit' in self.dfor.columns:
            pits  = self.dfor['pit'].copy()
        else:      
            pits = self.forecasted.compute_pit()        
        
        # Compute the ecdf on the pits
        ecdf = ECDF(pits)
        
        # Data work (note that the pit are computed by default)
        support = np.arange(0, 1, 0.01)
        
        # Fit it on the support
        pit_line = ecdf(support)
        uniform_line = stats.uniform.cdf(support)

        # Confidence intervals based on Rossi and Shekopysan JoE 2019
        confidence_interval = pd.DataFrame({
            'part of distribution': ['whole Distribution', 'Left Tail', 'Left Half', 'Right Half', 'Right Tail', 'Center', 'Tails'],
            'Interval': ['[0; 0:25]', '[0; 0:25]', '[0; 0:50]', '[0:50; 1]', '[0:75; 1]', '[0:25; 0:75]', '[0; 0:25] + [0:75; 1]'],
            0.01: [1.61, 1.24, 1.54, 1.53, 1.24, 1.61, 1.33],
            0.05: [1.34, 1.00, 1.26, 1.25, 1.00, 1.33, 1.10],
            0.10: [1.21, 0.88, 1.12, 1.12, 0.88, 1.19, 0.99]
        }).set_index('part of distribution')

        available_thresholds = [0.01, 0.05, 0.1] 
        available_part_distrib = confidence_interval.index
        assert threshold in available_thresholds, f'Test only available on {available_thresholds}'
        assert part_distribution in available_part_distrib, f"part_distribution should belong to one of the following options {available_part_distrib}"

        kp = confidence_interval.loc[part_distribution, threshold]/np.sqrt(len(pits))          
        ci_u = [x+kp for x in support]
        ci_l = [x-kp for x in support]

        fig, ax = plt.subplots(1, 1)
        ax.plot(support, pit_line, color='blue', label='Empirical CDF', lw=2)
        ax.plot(support, uniform_line, color='red', label='Theoretical CDF')
        ax.plot(support, ci_u, color='red', label=f'{int(threshold*100)}% critical values', linestyle='dashed')
        ax.plot(support, ci_l, color='red', linestyle='dashed')
        ax.legend(frameon=False)
        ax.set_xlabel(xlabel, labelpad=20)
        ax.set_ylabel(ylabel, labelpad=20)
        ax.set_title(title, y=1.02)
        return(fig)                
    
    # Plot the fxi rule as a pdf
    def plot_pdf_rule(self, fdate=None, q_low=0.05, q_high=0.95,
                      title=None, 
                      xlabel='bps', ylabel='density',
                      sample_lim=0.1, ax=None):
        """ 
        Summary chart of FX intervention rule

        fdate: str or None, default None
            Date to condition the data for the forecast. 
            By default, last date available

        q_low, q_high: float, between 0 and 1
            The quantiles of interest, on the left and right

        I should recode it to use directly the pdf instead of the sample...
        Not difficult, but I am tired now and I need to finish in a hurry


        """

        
        if fdate:
            assert fdate in self.dfor.index, "Date not in data"
        else: # Take the last date available
            fdate = self.dfor.index[-1].strftime('%Y-%m-%d')
        
        # Compute the pdf
        mu = self.dfor.loc[fdate, 'cond_mean']
        sigma = self.dfor.loc[fdate, 'cond_vol']
        y_cdf = lambda v: self.y_cdf(v, mu, sigma)
        y_pdf = lambda v: self.y_pdf(v, mu, sigma)
        y_ppf = lambda v: self.y_ppf(v, mu, sigma)
        sample_lim = 0.001
        support = np.linspace(y_ppf(sample_lim), y_ppf(1-sample_lim), 1000)
        
        pdf = y_pdf(support)
        
        # Compute the quantiles to determine the intervention region
        qval_low = y_ppf(q_low)
        qval_pdf_low = y_pdf(qval_low)
        qval_high = y_ppf(q_high)
        qval_pdf_high = y_pdf(qval_high)

        # Compute the mode
        x_mode = support[list(pdf).index(max(pdf))]
        y_mode = y_pdf(x_mode)

        
        # Prepare the plot        
        fig, ax = plt.subplots(1,1)
        ax.plot(support, pdf, lw=3)

        # Intervention regions
        ax.fill_between(support, 0, pdf, where=support<=qval_low, color='red',
                        label='Intervention region')
        ax.fill_between(support, 0, pdf, where=support>qval_high, color='red')

        #### Add text and lines about VaR and Mode
        # Low quantile
        ax.text(qval_low, qval_pdf_low , f'VaR {100*q_low}%', color='darkred',
                horizontalalignment='right', verticalalignment='bottom')
        ax.text(0.99*qval_low, 0, '{:.1f}'.format(qval_low),
                horizontalalignment='left', color='darkred',
                verticalalignment='top')
        ax.vlines(qval_low, 0, qval_pdf_low, linestyle='--')

        # High quantile
        ax.text(qval_high, qval_pdf_high , f'VaR {100*q_high}%',
                color='darkred',
                horizontalalignment='left', verticalalignment='bottom')
        ax.text(0.99*qval_high, 0, '{:.1f}'.format(qval_high),
                horizontalalignment='right', color='darkred',
                verticalalignment='top')
        ax.vlines(qval_high, 0, qval_pdf_high, linestyle='--')

        # Mode
        ax.text(x_mode, y_mode , f'Mode', color='darkred',
                horizontalalignment='center', verticalalignment='bottom')
        ax.text(x_mode, 0, '{:.1f}'.format(x_mode),
                horizontalalignment='center', color='darkred',
                verticalalignment='top')
        ax.vlines(x_mode, 1.1*min(pdf), y_mode, linestyle='--')

        
        ax.set_xlabel(xlabel, labelpad=20) # X axis data label
        ax.set_ylabel(ylabel, labelpad=20) # Y axis data label

        # Final customizations
        plt.xlim(min(support), max(support))
        plt.legend(loc='upper left', frameon=False, fontsize='x-small', handlelength=1)

        if title:
            plt.title(title, y=1.02)
        else:
            ttl = (f'Conditional density and intervention rule'
                   f' based on {fdate} information')
            plt.title(ttl, y=1.02)

        if ax:
            return(ax)
        else:
            return(fig)


    # VaR exceedance
    def plot_var_exceedance(self, qv_l=[0.025, 0.975], 
                            title_1='Log Returns and VaR Exceedance', 
                            title_2='Level',
                            swap_color=None, 
                            y1_label='',
                            y2_label='',
                            size=100):
        
        """ 
        Plot the VaR Exceedance 
        On the dependent variable, also possible to indicate level

        qv_l: list of two quantiles, 
              List of the upper and below quantiles
 

        """

        qv_l = sorted(qv_l)
        qv_labels_l = [f'cond_quant_{x:g}' for x in qv_l]

        
        # Prepare the frame
        dv = self.forecasted.VaR_FXI(qv_l)
        
        # Subselect the frame and plot it
        dvs = dv.loc[self.start_date:, :].copy()

        # Prepare the plot
        if self.level:
            fig, (ax1, ax2) = plt.subplots(2,1)
        else:
            fig, ax1 = plt.subplots(1,1)

        x = np.arange(len(dvs.index))

        if swap_color:
            ctop = 'green'
            cdown = 'red'
        else:
            ctop = 'red'
            cdown = 'green'
        
        # First plot
        ax1.plot(x, dvs[self.depvar], lw=3)
        ax1.scatter(x, dvs['Below'], alpha=0.8, c=ctop, marker='D', s=size,
                    label=f'Below VaR {qv_labels_l[0]}%')
        ax1.scatter(x, dvs['Above'], alpha=0.8, c=cdown, marker='o', s=size,
                    label=f'Above VaR {qv_labels_l[1]}%')
        equidate_ax(fig, ax1, dvs.index) # Adjust the dates
        ax1.set_title(title_1, y=1.02)
        ax1.set_xlabel('')
        ax1.set_ylabel(y1_label, labelpad=20)

        if self.level:
            # Second plot
            ax2.plot(x, dvs[self.level], lw=3)
            ax2.scatter(x, dvs['Level Below'], alpha=0.8, s=size,
                        c=ctop, marker='D', label='Level Below')
            ax2.scatter(x, dvs['Level Above'], alpha=0.8, s=size,
                        c=cdown, marker='o', label='Level Above')
            equidate_ax(fig, ax2, dvs.index) # Adjust the dates
            ax2.set_title(title_2, y=1.02)
            ax2.set_xlabel('')
            ax2.set_ylabel(y2_label, labelpad=20)

        plt.subplots_adjust(hspace=0.3)
        return(fig)


    def plot_fixed_exceedance(self, thresholds_t=(-200, 200), 
                              title_1='Log Returns and Fixed Thresholds '
                              'Exceedance', 
                              title_2='FX Level',
                              swap_color=None, 
                              y1_label='',
                              y2_label='',
                              size=100, hspace=0.3):
        
        """ 
        Plot the Fixed Thresholds Exceedance 
        On the dependent variable, also possible to indicate level

        thresholds_l: tuple of fixed thresholds for FXI, 
              
        """ 

        # Generate the data frame
        dv = self.forecasted.fixed_thresholds_FXI(thresholds_t)

        # Subselect the frame and plot it
        dvs = dv.loc[self.start_date:, :].copy()

        # Prepare the plot
        fig, (ax1, ax2) = plt.subplots(2,1)

        x = np.arange(len(dvs.index))

        # Color management
        if swap_color:
            ctop = 'green'
            cdown = 'red'
        else:
            ctop = 'red'
            cdown = 'green'

        # First plot
        ax1.plot(x, dvs[self.depvar], lw=3)
        ax1.scatter(x, dvs['Logret Below'], alpha=0.8, c=ctop,
                    marker='D', s=size)
        ax1.scatter(x, dvs['Logret Above'], alpha=0.8, c=cdown,
                    marker='o', s=size)
        equidate_ax(fig, ax1, dvs.index) # Adjust the dates
        ax1.set_title(title_1, y=1.02)
        ax1.set_xlabel('')
        ax1.set_ylabel(y1_label, labelpad=20)


        # Second plot
        ax2.plot(x, dvs[self.level], lw=3)
        ax2.scatter(x, dvs['Level Below'], alpha=0.8, s=size,
                    c=ctop, marker='D', label='Level Below')
        ax2.scatter(x, dvs['Level Above'], alpha=0.8, s=size,
                    c=cdown, marker='o', label='Level Above')
        equidate_ax(fig, ax2, dvs.index) # Adjust the dates
        ax2.set_title(title_2, y=1.02)
        ax2.set_xlabel('')
        ax2.set_ylabel(y2_label, labelpad=20)

        plt.subplots_adjust(hspace=hspace)

        plt.show()


    # Plot of the conditional cdf
    def plot_conditional_cdf(self, q_low=0.05, q_high=0.95,
                             title=('Conditional cumulative distribution'
                                    ' function and intervention thresholds'),
                             ylabel='quantile',
                             thresholds_t=None, 
                             swap_color=None,
                             size=100, 
                             xticks_freq=None):

        """ 
        Plot the conditional cdf over time and the intervention thresholds 
        I should recode it to use directly the cdf instead of the sample...
        Not difficult, but I am tired now and I need to finish in a hurry

        xticks_freq: None or int, default None
            Frequency of xticks: e.g., 2 is 1 out of 2. None for plt default

        """

        # Data work        
        dcq = self.dfor[self.cond_quant_labels_l].dropna().copy()
        dcq.columns = [x.replace('cond_quant_', '') for x in dcq.columns]

        dates_l = dcq.index[:-1] # Don't take the last one

        dclose = pd.DataFrame(index=dates_l,
                              columns=['realization','cond_quant'])

        # Find the closest quantile in the list
        for fdate in dates_l:
            realization = self.df.loc[fdate, self.depvar]
            cond_quant_l = dcq.loc[fdate, :]
            closest_quantile_idx = closest(realization, cond_quant_l)[0]
            closest_quantile = cond_quant_l.index[closest_quantile_idx]
            dclose.loc[fdate, :] = [realization, closest_quantile]

        dclose['Quantile'] = 100*(dclose['cond_quant'].astype(float))

        
        if thresholds_t:
            # Add the fixed thresholds
            dv = self.fixed_thresholds_FXI(thresholds_t)
            dclose = dclose.merge(dv[['FXI']],
                                  left_index=True, right_index=True)

            da = dclose.loc[dclose['FXI']=='Above', :].copy()
            db = dclose.loc[dclose['FXI']=='Below', :].copy()
            
        else:
            dv = self.forecasted.VaR_FXI(qv_l=[q_low, q_high])
            dclose = dclose.merge(dv[['FXI']],
                                  left_index=True, right_index=True)
            da = dclose.loc[dclose['FXI']=='Above', :].copy()
            db = dclose.loc[dclose['FXI']=='Below', :].copy()
            
        # Prepare the plot
        fig, ax = plt.subplots()
        ax.plot(dclose.index, dclose['Quantile'], lw=3)

        # Scatter lines with fixed thresholds
        if swap_color:
            ctop = 'green'
            cdown = 'red'
        else:
            ctop = 'red'
            cdown = 'green'


        ax.scatter(da.index, da['Quantile'], alpha=0.8, c=ctop,
                   marker='D', s=size)
        ax.scatter(db.index, db['Quantile'], alpha=0.8, c=cdown,
                   marker='o', s=size)
            

        # Horizontal lines
        ax.axhline(y=100*q_low, color='red', linestyle='--', lw=2)
        ax.axhline(y=100*q_high, color='red', linestyle='--', lw=2)
        ax.axhline(y=25, color='blue', linestyle='-', lw=1)
        ax.axhline(y=50, color='black', linestyle='-', lw=1)
        ax.axhline(y=75, color='blue', linestyle='-', lw=1)

        # Add the ticks, if needed
        new_t_l = [100*q_low, 25, 50, 75, 100*q_high]
        #new_ticks_l = sorted(list(ax.get_yticks()) + new_t_l)
        new_ticks_l = new_t_l
        extra_idx_l = [new_ticks_l.index(x) for x in new_t_l]
        ax.set_yticks(new_ticks_l) # Add new ticks

        for idx in extra_idx_l: 
            ax.get_yticklabels()[idx].set_color("darkred")            

        ax.set_title(title, y=1.02)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, labelpad=20)

        # Manage frequency of xticks & make sure the last one always visible
        if xticks_freq:
            start, end = ax.get_xlim()
            t_seq = np.append(np.arange(start, end-5, xticks_freq), end)
            ax.xaxis.set_ticks(t_seq)

        #ax.tick_params(direction='out', pad=20)
        ax.tick_params(axis='x', rotation=45, pad=20)
        plt.ylim(0,100)

        return(fig)

    # Fan chart
    def plot_fan_chart(self,
                       title=None,
                       ylabel='',
                       xticks_freq=None):
        """ 
        Plot a fan chart of the conditional density
        Colors focus on the extreme points instead than the median

        xticks_freq: None or int, default None
            Frequency of xticks: e.g., 2 is 1 out of 2. None for plt default

        """

        # Select the quantiles
        qfc_l = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        qfc_labels_l = [f'cond_quant_{x}' for x in qfc_l]

        dfc = self.dfor[['true_val'] + qfc_labels_l].copy()

        dfc.columns = [x.replace('cond_quant_', '') for x in dfc.columns]


        # Prepare the plot
        fig, ax = plt.subplots(1, 1)
        
        # Plot each quantile values
        ax.plot(dfc['0.05'], linestyle=':', color='black', label='5th')
        ax.plot(dfc['0.1'], linestyle='-.', color='black', label='10th')
        ax.plot(dfc['0.25'], linestyle='--', color='black', label='25th')
        ax.plot(dfc['0.5'], linestyle='-', color='black', label='Median',
                lw=2)
        ax.plot(dfc['0.75'], linestyle='--', color='black', label='75th')
        ax.plot(dfc['0.9'], linestyle='-.', color='black', label='90th')
        ax.plot(dfc['0.95'], linestyle=':', color='black', label='95th')

        # Fill the colors between the lines with different transparency level
        ax.fill_between(dfc.index, dfc['0.05'], dfc['0.1'],
                        color='darkred', alpha=0.75)
        ax.fill_between(dfc.index, dfc['0.9'], dfc['0.95'],
                        color='darkred', alpha=0.75)
        ax.fill_between(dfc.index, dfc['0.1'], dfc['0.25'],
                        color='red', alpha=0.4)
        ax.fill_between(dfc.index, dfc['0.75'], dfc['0.9'],
                        color='red', alpha=0.4)
        ax.fill_between(dfc.index, dfc['0.25'], dfc['0.5'],
                        color='blue', alpha=0.15)
        ax.fill_between(dfc.index, dfc['0.5'], dfc['0.75'],
                        color='blue', alpha=0.15)  

        if title==None:
            t1 = f'Fan chart of predictive {self.depvar}'
            ttl = t1 + '\n 1, 5, 10, 25, 50, 75, 90, 95 Conditional Quantiles'
        else:
            ttl=title
        ax.set_title(ttl, y=1.02)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel, labelpad=20)

        # Manage frequency of xticks & make sure the last one always visible
        if xticks_freq:
            start, end = ax.get_xlim()
            t_seq = np.append(np.arange(start, end-5, xticks_freq), end)            
            ax.xaxis.set_ticks(t_seq)

        ax.tick_params(axis='x', rotation=45)

                
        return(fig)
