# -*- coding: utf-8 -*-
"""
VaR FXI model: Application to Mexico
Romain Lafarguette 2020, rlafarguette@imf.org
Time-stamp: "2020-10-15 00:40:52 Romain"
"""

###############################################################################
#%% Modules
###############################################################################
# System paths
import os, sys
sys.path.append(os.path.abspath('modules'))

# Global modules
import importlib                                        # Operating system
import pandas as pd                                     # Dataframes
import numpy as np                                      # Numeric Python
import datetime                                         # Dates
import arch                                             # ARCH/GARCH models

# Functional imports
from datetime import datetime as date                   # Short date function
from dateutil.relativedelta import relativedelta        # Dates manipulation 
from statsmodels.distributions.empirical_distribution import ECDF

# ARCH package functional imports
from arch.univariate import ARX # Drift model
from arch.univariate import (ARCH, GARCH, EGARCH, EWMAVariance, # Vol process
                             FixedVariance, RiskMetrics2006) 
from arch.univariate import (Normal, StudentsT, # Distribution of residuals
                             SkewStudent, GeneralizedError)

# Ignore a certain type of warnings which occurs in ML estimation
from arch.utility.exceptions import (
    ConvergenceWarning, DataScaleWarning, StartingValueWarning,
    convergence_warning, data_scale_warning, starting_value_warning,
)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Local modules
import distGARCH; importlib.reload(distGARCH)           # Distributional GARCH
from distGARCH import DistGARCH

# Graphics
import matplotlib.pyplot as plt                         # Graphical package  
import seaborn as sns                                   # Graphical tools

# Graphics options
plt.rcParams["figure.figsize"] = 25,15
sns.set(style='white', font_scale=2, palette='deep', font='Arial',
        rc={'text.usetex' : False}) 

# Pandas options
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)

###############################################################################
#%% Local functions
###############################################################################
def logret(series): return(np.log(series/series.shift(1)))

###############################################################################
#%% Data loading and cleaning
###############################################################################
macro_p = os.path.join('data', 'macro_data.csv')
dm = pd.read_csv(macro_p, parse_dates=['date'], index_col=['date'])

inter_p = os.path.join('data', 'intervention_data.csv')
di = pd.read_csv(inter_p, parse_dates=['date'], index_col=['date'])

df = pd.merge(dm, di, on=['date'], how='left').sort_index().copy() # Merge
df = df[~df.index.duplicated()].copy() # Duplicated index

# New macro variables
df['FX level'] = df['mxn_usd_spot'].copy()
df['FX log returns'] = 10000*logret(df['mxn_usd_spot'])
df['Bid-ask spread abs value'] = np.abs(df['bid_ask_spread'])
df['Min-max spread abs value'] = np.abs(df['min_max_spread'])
df['Forward points first difference'] = df['mxn_fwd_1m'].diff(1)/100
df['Interbank rate vs Libor'] = (df['mxn_interbank_1m']
                                 - df['usa_libor_1m']).diff(1)
df['VIX first diff'] = df['vix'].diff(1)
df['EURUSD log returns'] = 10000*logret(df['eur_usd_spot'])
df['Oil prices log returns'] = 10000*logret(df['oil_prices'])

# FX intervention variables
df['FX intervention in USD'] = df['sell_amount'].fillna(0)
df['fx_intervention_minprice'] = df.loc[df['type']=='min price',
                                        'sell_amount'].fillna(0)
df['fx_intervention_nominprice'] = df.loc[df['type']=='no min price',
                                          'sell_amount'].fillna(0)
df['FX intervention dummy'] = 0
df.loc[df['FX intervention in USD'] > 0, 'FX intervention dummy'] = 1
df['FX intervention dummy lag'] = df['FX intervention dummy'].shift(1)
df['Intercept'] = 1

###############################################################################
#%% Fit the GARCH model for different specifications
###############################################################################
# Prepare the list of variables
microstructure = ['Bid-ask spread abs value',
                  'Min-max spread abs value',
                  'Forward points first difference']

cip = microstructure + ['Interbank rate vs Libor']

eurusd = cip + ['EURUSD log returns']

vix = eurusd + ['VIX first diff']

baseline = vix + ['FX intervention dummy lag', 'Oil prices log returns']

# List of models
models_l = [microstructure, cip, eurusd, vix, baseline]
labels_l = ['Microstructure', 'CIP', 'Dollar move', 'Risk Appetite', 'Baseline']

specification_tables_l = list()
specification_tables_short_l = list()
error_l = list()
for label, model in zip(labels_l, models_l): # Run for different specifications
    try:
        dgm = DistGARCH(depvar_str='FX log returns',
                        data=df,
                        level_str='FX level', 
                        exog_l=model, 
                        lags_l=[1], 
                        vol_model=EGARCH(1,1,1),
                        dist_family=SkewStudent())

        # Fit the model
        dgfit = dgm.fit()

        # Generate the tables
        var_d = {'FX l...rns[1]':'Lag FX log returns'}
        sumtable = dgfit.summary_table(model_name=label, var_d=var_d,
                                       print_pval=True)
        sumtable_short = dgfit.summary_table(model_name=label, var_d=var_d,
                                             print_pval=False)
        specification_tables_l.append(sumtable)
        specification_tables_short_l.append(sumtable_short)
    except:
        error_l.append(label)

print(error_l)

###############################################################################
#%% Export the table
###############################################################################
# Merge all the summary tables (need to reorder some rows)
dsum = pd.concat(specification_tables_l, axis=1)
dsum_short = pd.concat(specification_tables_short_l, axis=1)

new_index = ['Intercept', 'Lag FX log returns',
             'Bid-ask spread abs value', 'Min-max spread abs value',
             'Forward points first difference',
             'Interbank rate vs Libor', 
             'EURUSD log returns', 'VIX first diff', 
             'FX intervention dummy lag', 'Oil prices log returns', 
             'Omega', 'Alpha', 'Gamma', 'Beta', 'Nu', 'Lambda', 
             'R2', 'R2 adjusted', 'Number of observations', 
             'Pvalue in parenthesis', 'Significance *10%, **5%, ***1%']

avl_x = [x for x in new_index if x in dsum.index]
dsumf = dsum.loc[avl_x, :].fillna('')

# Export to latex
tex_f = os.path.join('output', 'regressions_table.tex')
dsumf.fillna('').to_latex(tex_f)

# Export the frame to LateX without pvalues for beamer (shorter tables)
short_new_index = [x for x in new_index if x in dsum_short.index]
dsum_short_f = dsum_short.loc[short_new_index, :].copy()
tex_short_f = os.path.join('output', 'regressions_table_short.tex')
dsum_short_f.fillna('').to_latex(tex_short_f)
dsum_short_f

###############################################################################
#%% Baseline model: Fit and forecast
###############################################################################

#### Specify the model
dg = DistGARCH(depvar_str='FX log returns',
               data=df,
               level_str='FX level', 
               exog_l=baseline, # Defined above 
               lags_l=[1], 
               vol_model=EGARCH(1,1,1),
               # ARCH(1,1), EGARCH(1,1,1), GARCH(1,1),
               # EWMAVariance(None), RiskMetrics2006(),
               dist_family=SkewStudent(),
               # Normal(), StudentsT(), SkewStudent(), GeneralizedError()
)

# Fit the model
dgf = dg.fit()

# Forecast 2020
dgfor = dgf.forecast('2020-01-01', horizon=1)

###############################################################################
#%% Plot
###############################################################################
# Plot
dgfor.pit_plot(title=
               'Probability Integral Transform (PIT) Test, Out-of-sample')

# Save the figure
pitchart_f = os.path.join('output', 'pitchart.pdf')
plt.savefig(pitchart_f, bbox_inches='tight')
plt.show()
plt.close('all')



#%%
# Plot
dgfor.plot_pdf_rule(fdate='2020-04-03', q_low=0.025, q_high=0.975)

# Save the figure
var_rule_f = os.path.join('output', 'var_rule.pdf')
plt.savefig(var_rule_f, bbox_inches='tight')
plt.show()
plt.close('all')

###############################################################################
#%% Density performance
###############################################################################
start_date = '2020-01-01'
hist_sample = df.loc[:start_date, 'FX log returns'].dropna().values
fdate_l = list(df.loc[start_date:, 'FX log returns'].index)[1:]

###############################################################################
#%% Logscore
###############################################################################
forecast_date_l = list() # because some dates have no pdf
logscore_l = list()

for fdate in fdate_l:
    try:
        true_val = float(df.loc[fdate, 'FX log returns'])
        log_score = np.log(dgfor.dist_fit(fdate).pdf(true_val))
        logscore_l.append(log_score)
        forecast_date_l.append(fdate)
    except:
        print(fdate)

forecast_sample = df.loc[forecast_date_l, 'FX log returns'].dropna().values

###############################################################################
#%% Unconditional Distribution Benchmarking
###############################################################################
# Fit the unconditional distribution with Gaussian Kernel
from scipy import stats
unc_kde = stats.gaussian_kde(hist_sample)
unc_logscore = np.log(unc_kde.evaluate(forecast_sample))

# Estimate the PIT
line_support = np.arange(0,1, 0.01)
unc_pits = [unc_kde.integrate_box_1d(np.NINF, x) for x in forecast_sample]

# Compute the ecdf on the pits
unc_ecdf = ECDF(unc_pits)
# Fit it on the line support
unc_pit_line = unc_ecdf(line_support)

# Confidence intervals based on Rossi and Shekopysan JoE 2019
ci_u = [x+1.34*len(unc_pits)**(-0.5) for x in line_support]
ci_l = [x-1.34*len(unc_pits)**(-0.5) for x in line_support]

# Prepare the plots
fig, ax = plt.subplots(1)

ax.plot(line_support, unc_pit_line, color='blue',
        label='Out-of-sample empirical CDF',
        lw=2)
ax.plot(line_support, line_support, color='red', label='Theoretical CDF')
ax.plot(line_support, ci_u, color='red', label='5 percent critical values',
        linestyle='dashed')
ax.plot(line_support, ci_l, color='red', linestyle='dashed')
ax.legend()
ax.set_xlabel('Quantiles', labelpad=20)
ax.set_ylabel('Cumulative probability', labelpad=20)
ax.set_title('Unconditional Distribution PIT test', y=1.02)
plt.show()

###############################################################################
#%% Kernel regressions
###############################################################################



###############################################################################
#%% Log score comparisons via Diebold Mariano test statistic
###############################################################################
model_ls_diff = logscore_l - unc_logscore
norm_factor = np.sqrt(np.var(model_ls_diff)/len(logscore_l))

tt = np.mean(model_ls_diff)/norm_factor # Follows a N(0,1)
pval = 1-stats.norm.cdf(tt, 0, 1) # Two-sided test
print(f'test statistic: {tt:.3f}, pval:{pval:.3f}')



###############################################################################
#%% 
###############################################################################












