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

# ARCH package functional imports
from arch.univariate import ARX # Drift model
from arch.univariate import GARCH, EGARCH, EWMAVariance # Volatility models
from arch.univariate import Normal, StudentsT, SkewStudent # Distributions

# Other functional imports
from collections import namedtuple                      # Named tuples
from scipy import stats                                 # Statistical tools 
from statsmodels.distributions.empirical_distribution import ECDF

# Graphics
# NB: for some systems, need to remove matplotlib.use('TkAgg')
import matplotlib                                       # Graphical package
matplotlib.use('TkAgg') # Must be called before importing plt
import matplotlib.pyplot as plt
from matplotlib import cm                               # Colormaps
import matplotlib.colors as mcolors
import seaborn as sns                                   # Graphical package

# Local modules
import joyplot2; importlib.reload(joyplot2)
from joyplot2 import joyplot2

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

    Output
    ------
    A GARCH class object, wrapping .fit() and .fit().forecast() classes

    Usage:
    dg = DistGARCH(depvar_str, level_str, data)

    """
    __description = "Distributional GARCH"
    __author = "Romain Lafarguette, IMF/MCM, rlafarguette@imf.org"

    # Initializer
    def __init__(self, depvar_str, data, level_str=None, 
                 exog_l=None, lags_l=[1],
                 vol_model=GARCH(1,1),
                 dist_family=Normal(),
                 random_state=42, # Answer to the Ultimate Question of Life 
    ):
        
        # Attributes
        self.depvar = depvar_str
        self.level = level_str
        self.exog_l = exog_l
        self.lags_l = lags_l
        self.rs = np.random.RandomState(random_state) 
        
        # Special attributes
        self.vol_model = vol_model # By default GARCH(1,1)
        
        self.dist_family = dist_family # By default Normal distribution
        self.dist_family._random_state = self.rs
        
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
                       lags=self.lags_l)

        # Volatiliy model
        self.mod.volatility = self.vol_model

        # Specify the distribution of the error terms
        self.mod.distribution = self.dist_family 
        
    # Class-methods (methods which returns a class defined below)    
    def fit(self, cov_type='robust', disp='off', update_freq=1):
        return(DistGARCHFit(self, cov_type, disp, update_freq))

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
        ax3 = sns.distplot(data[self.depvar])
        ax3.set_title(title_density, y=1.02)
        ax3.set_xlabel(y_label_returns)
        
        # Adjust
        plt.subplots_adjust(hspace=0.5)

        # Exit
        return(fig)
       
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
    def __init__(self, DistGARCH, cov_type, disp, update_freq):

        self.__dict__.update(DistGARCH.__dict__) # Import all attributes

        # Fit the model (just take from the ARCH model)    
        self.res = self.mod.fit(
            cov_type=cov_type, 
            disp=disp, 
            update_freq=update_freq)

        # Print the summary at each fit
        print(self.res.summary())

        # Erros Distribution parameters (depends on the distribution)
        # Note that the distributions are NORMALIZED
        if self.mod.distribution.name=='Normal':
            self.dist_params = np.array([0,1]) # By default
            self.scipy_dist = scipy.stats.norm
            
        elif self.mod.distribution.name=="Standardized Student's t":
            self.dist_params = self.res.params['nu']
            self.scipy_dist = scipy.stats.t
            
        elif self.mod.distribution.name=="Standardized Skew Student's t":
            self.dist_params = self.res.params[['nu', 'lambda']]
            self.scipy_dist = scipy.stats.nct
            
        elif self.mod.distribution.name=="Generalized Error Distribution":
            self.dist_params = self.res.params['nu']
            self.scipy_dist = scipy.stats.gennorm
                                    
        else:
            raise ValueError('Distribution name mis-specified')


        # Compute the historical and in-sample volatility and mean
        self.hist_avg = self.df[self.depvar].mean()
        self.hist_vol = self.df[self.depvar].std()
        self.in_cond_vol = self.res.conditional_volatility
               
    # Class-methods (methods which returns a class defined below)    
    def forecast(self, start_date, horizon=1, fmethod='analytic',
                 sample_size=10000):
        return(DistGARCHForecast(self, start_date, horizon, fmethod,
                                 sample_size))

    # Public methods: Shocks Simulation
    def shock_simulate(self, nobs=100, burn=0, mean_mult=1, vol_mult=1):
        """ 
        Simulate the model based on estimated parameters 

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
    def summary_table(self, model_name='GARCH model', var_d=None,
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
                    'nu':'Nu', 
                    'lambda':'Lambda'}

        if var_d:
            rename_d.update(var_d)

        summary_table = summary_table.rename(rename_d, axis='index')
        summary_table_nna = summary_table.fillna('').copy()

        return(summary_table_nna)
    
    # Public methods: Plots
    def summary_fit_plot(self):
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
                 sample_size):

        self.__dict__.update(DistGARCHFit.__dict__) # Import all attributes 
        
        self.start_date = start_date
        self.horizon = horizon
        self.fmethod = fmethod
        self.sample_size = sample_size

        # Construct the forecasted matrix of exogeneous regressors
        self.exog_fcast_d = dict() # Construct the dictionary of regressors

        if len(self.exog_l)>0:
            for var in self.exog_l:
                #val = self.df_exog.loc[self.start_date, [var]].values
                #self.exog_fcast_d[var] = np.full((len(self.df_exog), 1), val)
                self.exog_fcast_d[var] = self.df_exog.loc[:, [var]].values
                
        elif self.exog_l == None: # No exogeneous regressors
                self.exog_fcast_d = None
        else:
            raise ValueError('List of exogeneous regressors misspecified')
                
        # Run the forecasts from the arch package
        self.forecasts = self.res.forecast(horizon=self.horizon,
                                           x=self.exog_fcast_d, 
                                           start=self.start_date, 
                                           method=self.fmethod,
                                           reindex=True)
                                                        
        # Extract the forecasted conditional mean and variance
        self.cond_mean = self.forecasts.mean[start_date:]
        self.cond_var = self.forecasts.variance[start_date:]
        
        dfor = pd.merge(self.cond_mean, self.cond_var,
                        left_index=True, right_index=True)
        self.dfor = pd.DataFrame(dfor.values, index=self.cond_var.index,
                                 columns=['cond_mean', 'cond_var'])
        
        # Compute the normalized true values (for the PIT test)
        avl_idx_l = [x for x in self.df.index if x in self.dfor.index]
        self.dfor.insert(0, 'true_val',
                         self.df.loc[avl_idx_l, self.depvar])
        self.dfor['cond_vol'] = np.sqrt(self.dfor['cond_var'])

        # Because the distribution are on the normalized values
        self.dfor['norm_true_val'] = ((self.dfor['true_val']
                                      - self.dfor['cond_mean'])
                                      /self.dfor['cond_vol'])
        
        #### CDF and sampling estimation
        # Important: the standardized distribution does NOT change over time
        # Parameters are fixed, The conditional mean and volatility change
        q_l = [0.01] + [round(x,3) for x in np.arange(0.025,1,0.025)] + [0.99]
        self.cond_quant_labels_l = [f"cond_quant_{q:g}" for q in q_l]
        
        # Define functions associated with conditional distribution 
        if self.mod.distribution.name=='Normal': # No parameters
            self.cond_ppf = lambda q: self.mod.distribution.ppf(q)
            self.cond_cdf = lambda v: self.mod.distribution.cdf(v)
            
        else:
            self.cond_ppf = lambda q: self.mod.distribution.ppf(q,
                                                            self.dist_params)
            self.cond_cdf = lambda v: self.mod.distribution.cdf(v,
                                                            self.dist_params)
            
        err_cond_quant = self.cond_ppf(q_l)
        self.dfor['pit'] = self.cond_cdf(self.dfor['norm_true_val'])
        self.dfor['norm_true_val_cdf'] = self.dfor['pit'].copy() # Simple name
        
        self.mod.distribution._random_state = self.rs
        err_sampler = self.mod.distribution.simulate(self.dist_params)
        
        # PAY ATTENTION THAT THE DISTRIBUTIONS ARE STANDARDIZED
        # TO RECOVER THE SERIES, NEED TO REINFLATE WITH MEAN AND VAR
        
        # Summarize the conditional quantiles in a dataframe
        # xbar = (x-mu)/std => x = mu + xbar*std
        # mean + sqrt(cond_var)*the quantiles of the errors terms
        for var in self.cond_quant_labels_l: self.dfor[var] = np.nan
        self.dfor[self.cond_quant_labels_l] = (self.dfor[['cond_mean']].values
                                            + (self.dfor[['cond_vol']].values
                                                  * err_cond_quant[None, :]))
                      
        # Sampling the error terms and deriving the values for y in a dataframe
        err_sample = err_sampler(self.sample_size)
        self.sample = self.cond_mean.values +(np.sqrt(self.cond_var).values
                                                  * err_sample)
        self.sample = pd.DataFrame(self.sample,
                                   columns=range(self.sample_size),
                                   index=self.dfor.index)

    # Public methods
    def dist_fit(self, fdate):
        """
        Return the scipy random variable for a given date
        """
        
        # Take a sample at a given date
        sample = self.sample.dropna()

        if fdate:
            assert fdate in sample.index, "Date not in data sample"
            ssample = sample.loc[fdate, :].values
        else: # Take the last date available
            ssample = sample.tail(1).values
            fdate = sample.tail(1).index[0].strftime('%Y-%m-%d')

        # Fit the distribution
        params_fit = self.scipy_dist.fit(ssample)
        rv = self.scipy_dist(*params_fit) # Frozen random variate

        # Return the scipy random variable
        return(rv)
        
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
    def joyplot_out(self,
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

        # Take a sample at a given date
        sample = self.sample.dropna()

        if fdate:
            assert fdate in sample.index, "Date not in data sample"
            ssample = sample.loc[fdate, :].values
        else: # Take the last date available
            ssample = sample.tail(1).values
            fdate = sample.tail(1).index[0].strftime('%Y-%m-%d')

        # Fit the distribution
        # I am lazy, the arch dist have no pdf readily done so I fit a sample..
        params_fit = self.scipy_dist.fit(ssample)
        rv = self.scipy_dist(*params_fit) # Frozen random variate
        support = np.linspace(np.percentile(ssample, sample_lim),
                              np.percentile(ssample, 100-sample_lim), 1000)

        # Compute the pdf
        pdf = rv.pdf(support)
        #pdf = self.scipy_dist.pdf(support)
        
        # Compute the quantiles to determine the intervention region
        qval_low = rv.ppf(q_low)
        qval_pdf_low = rv.pdf(qval_low)
        qval_high = rv.ppf(q_high)
        qval_pdf_high = rv.pdf(qval_high)

        # Compute the mode
        x_mode = support[list(pdf).index(max(pdf))]
        y_mode = rv.pdf(x_mode)

        
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
        plt.legend()

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
        dv = self.VaR_FXI(qv_l)
        
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
        dv = self.fixed_thresholds_FXI(thresholds_t)

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
            dv = self.VaR_FXI(qv_l=[q_low, q_high])
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

    # Plot the probability integral transform test
    def pit_plot(self,
                 xlabel='Quantiles',
                 ylabel='Cumulative probability',
                 title=('Out-of-sample conditional density:'
                        ' Probability Integral Transform (PIT) test')):
        
        # Data work (note that the pit are computed by default)
        support = np.arange(0,1, 0.01)
        pits = self.dfor['pit'].dropna().copy()

        # Compute the ecdf on the pits
        ecdf = ECDF(pits)

        # Fit it on the support
        pit_line = ecdf(support)

        # Compute the KS statistics (in case of need)
        ks_stats = stats.kstest(self.dfor['pit'].dropna(), 'uniform')

        # Confidence intervals based on Rossi and Shekopysan JoE 2019
        ci_u = [x+1.61*len(pits)**(-0.5) for x in support]
        ci_l = [x-1.61*len(pits)**(-0.5) for x in support]

        #new_title = (f'{title} \n Kolmogorov-Smirnov Test of Uniformity')
        
        # Prepare the plots
        fig, ax = plt.subplots(1)
                
        ax.plot(support, pit_line, color='blue',
                label='Out-of-sample empirical CDF',
                lw=2)
        ax.plot(support, support, color='red', label='Theoretical CDF')
        ax.plot(support, ci_u, color='red', label='1 percent critical values',
                linestyle='dashed')
        ax.plot(support, ci_l, color='red', linestyle='dashed')
        ax.legend()
        ax.set_xlabel(xlabel, labelpad=20)
        ax.set_ylabel(ylabel, labelpad=20)
        ax.set_title(title, y=1.02)

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

