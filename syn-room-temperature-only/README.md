# Synthetic data studies for hERG paper I

Synthetic data studies for hERG kinetics at room temperature.

To reproduce the results in the supporting materials in the publication, run the scripts in the order as listed.


## Requirements

- myokit ([Myokit](http://myokit.org/))
- pints ([Pints](https://github.com/pints-team/pints))

Above libraries should be included/installed in [../lib/](../lib).


## Description

### Fitting
- [fit.py](./fit.py): Simple fitting for a cell; `Usage: python fit.py [int:cell_id] --optional [N_repeats]`
- [fit.sh](./fit.sh): Bash script calling [fit.py](./fit.py) as batches


### MCMC
- [mcmc.py](./mcmc.py): Run MCMC for one cell, usage: `python mcmc.py [cell-id]`
- [mcmc.sh](./mcmc.sh): Bash script to run MCMC for multiple cells by calling [mcmc.py](./mcmc.py)


### Hierarchical Bayesian Model (HBM)
- [hbm.py](./hbm.py): Run hierarchical Bayesian model (HBM) using Metropolis-Within-Gibbs (MWG)
- [pseudohbm.py](./pseudohbm.py): Run HBM using pseudo-MWG, requires individual cells' MCMC chain, and plot results


### Plots
- [plot-single-mcmc.py](./plot-single-mcmc.py): Plot single MCMC result as marginal histograms and traces
- [plot-pseudohbm-cov.py](./plot-pseudohbm-cov.py): Plot pseudo-MWG covariance results against true values
- [plot-hbm-v-pseudohbm.py](./plot-hbm-v-pseudohbm.py): Plot HBM results inferred using full MWG versus pseudo-MWG
- [plot-hbm-v-pseudo2hbm.py](./plot-hbm-v-pseudo2hbm.py): Plot HBM results inferred using full MWG versus _simplified_ pseudo-MWG
- [pseudohbm-convergence.py](./pseudohbm-convergence.py): Plot HBM pseudo-MWG convergence to true values
- [pseudo2hbm-convergence.py](./pseudo2hbm-convergence.py): Plot HBM _simplified_ pseudo-MWG convergence to true values

