Foreign Exchange Intervention Rules for Central Banks:  A Risk-Based Framework
================================

This notebook replicates the tables and the charts of the IMF WP on Foreign Exchange Intervention Rules for Central Banks.
It uses a Python package that I have written, DistGARCH, also available in this Github folder, with the public FX intervention data from the Banco Mexico. DistGARCH is based on the ARCH package of Kevin Sheppard.

You can use the code for non-commercial applications, providing that you cite the IMF Working Paper
Lafarguette, R. and Veyrune, R. (2020) Foreign Exchange Intervention Rules for Central Banks: A Risk-Based Framework, IMF Working Paper no 20XX

Author: Romain Lafarguette, August 2020
If you have any question, please contact me via Github or rlafarguette "at" imf "dot" org

The main files of interest are:
- The Jupyter Notebook "VaR-FX Interventions.ipynb" which contains the codes to replicate the paper
- The distGARCH.py file which contains the package used to estimate the GARCH and derive the conditional distribution
- joyplot2.py a file for producing joyplots, amended from https://github.com/sbebo/joypy (I will do a pull request soon)
- MXN_data.csv and intervention_data.csv contain the publicly available data

**Reuse  of this  tool and  IMF data  does not  imply any  endorsement of  the
research  and/or product.  Any research  presented should  not be  reported as
representing  the   views  of  the   IMF,  its  Executive  Board,   or  member
governments.**

Note that  the Github repo contains only publicly available data. 


