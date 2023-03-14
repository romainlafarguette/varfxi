VaR-Rule for FX Interventions
================================

Link to the Python notebook: https://github.com/romainlafarguette/VaR-FX-Interventions/blob/master/notebooks/VaR-FX%20Interventions.ipynb

The Python notebook replicates the tables and the charts of the IMF WP on
"Foreign Exchange Interventions Rules for Central Banks: A Risk-Based Framework"

**IMPORTANT: BECAUSE OF AN UPDATE OF THE ARCH PACKAGE AFTER 4.19, and in particular the
random number generator, the way the random seed is managed has changed. Some
results are therefore slightly different (e.g. the pdf plot) by a few pips as
in the IMF WP, but are qualitatively similar. The journal version will reflect
the new version**


The paper uses  a Python package  that I have  written, DistGARCH, also  available in
this  Github folder,  with  the public  FX intervention  data  from the  Banco
Mexico. DistGARCH is based on the ARCH package of Kevin Sheppard.

You can use the code for non-commercial applications, providing that you cite the IMF Working Paper
Lafarguette, R. and Veyrune, R. (2020) "Foreign Exchange Interventions Rules for Central Banks: A Risk-Based Framework", IMF Working Paper

The folder is organized as follows:
- mxn_estimation.py is the pure Python file with the core estimation and
  robustness analysis
- VaR-FX Interventions.ipynb is a Jupyter notebook, which illustrates the approach
- modules/ contains the modules for this project, in particular distGARCH which infers a conditional distribution from a GARCH model
- data/ contains public data files, with FX rate and FX interventions from Banco Mexico website
- img/ contains some images to illustrate the Jupyter Notebook

**Reuse of this  tool and  IMF data  does not  imply any  endorsement of  the research  and/or product.  Any research  presented should  not be  reported as
representing  the   views  of  the   IMF,  its  Executive  Board,   or  member
governments.**

Note that the Github repo contains only publicly available data. 

Author: Romain Lafarguette, August 2020

If you have any question, please contact me via Github or rlafarguette "at" imf "dot" org
