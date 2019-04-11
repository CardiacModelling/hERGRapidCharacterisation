# hERG paper I: using an automated high-throughput system

Study hERG kinetics at room temperature.

To reproduce the results in the publication, run the scripts in the order as listed.
All scripts under "Paper figures" should reproduce all figures in the publication.


## Requirements
- myokit ([Myokit](http://myokit.org/))
- pints ([Pints](https://github.com/pints-team/pints))

Above libraries should be included/installed in [../lib/](../lib).


## Description

### Fitting
- [fit-fromqc.py](./fit-fromqc.py): Fitting for all cells selected from [QC](../qc) (note later we have manual selections)


### Validation
- [plot-validation-protocols.py](./plot-validation-protocols.py): Plot fitting results with validation protocols (so far only APs)
- [plot-validation-protocolsi-zoom.py](./plot-validation-protocols-zoom.py): Plot fitting results with validation protocols with zoom in at spikes (so far only APs)
- [re-leak-correct.py](./re-leak-correct.py): Re-apply leak correction to the leak corrected recordings, as some of them are seemingly over-leak corrected
- [plot-rmsd-hist.py](./plot-rmsd-hist.py): Plot fitting and validation (normalised) RMSD histograms for all selected cells


### Selections
- [manualselected-herg25oc1.txt](./manualselected-herg25oc1.txt): Manual selection based on staircase ramp protocol recording only (after automated [QC](../qc) selection)
- [manualv2selected-herg25oc1.txt](./manualv2selected-herg25oc1.txt): Merged manual selection based on both [manualselected-herg25oc1.txt](./manualselected-herg25oc1.txt) and [manual-unselected-validation-herg25oc1.txt](./figs/manual-unselected-validation-herg25oc1.txt) which based on the validation protocol recordings
- [merge-selection-lists.py](./merge-selection-lists.py): Merge two selection (or with an unselection) lists


### MCMC
- [mcmc.py](./mcmc.py): Run MCMC for one cell, usage: `python mcmc.py [cell-id]`
- [mcmc.sh](./mcmc.sh): Bash script to run MCMC for multiple cells by calling [mcmc.py](./mcmc.py)


### Hierarchical Bayesian Model (HBM)
- [hbm.py](./hbm.py): Run hierarchical Bayesian model (HBM) using Metropolis-Within-Gibbs (MWG)
- [plot-hbm.py](./plot-hbm.py): Plot HBM results
- [test-simple-mean.py](./test-simple-mean.py): Compare HBM results with simple mean and covariance calculation, see if HBM is necessary!
- [pseudohbm.py](./pseudohbm.py): Run HBM using pseudo-MWG, requires individual cells' MCMC chain, and plot results


### Voltage error
- [voltage-error-gen-current.py](./voltage-error-gen-current.py): Generate synthetic data [fakedata-voltageoffset](./fakedata-voltageoffset) by assumimg `V_eff = V_command - V_err` where we estimate `V_err = expected_EK - est_EK`
- [voltage-error-fit.py](./voltage-error-fit.py): Fitting all synthetic data [fakedata-voltageoffset](./fakedata-voltageoffset) with no knowledge assumed about `V_eff = V_command - V_err`
- [voltage-error-plot-cov.py](./voltage-error-plot-cov.py): Plot covariance/correlation matrix with voltage error fit on top
- [voltage-error-plot-cov-2.py](./voltage-error-plot-cov-2.py): Plot pairwise individual parameters (from MCMC mean) with voltage error fit on top


### Paper figures
- [paper-rank-cells.py](./paper-rank-cells.py): Rank cells according to the hERG peak under hERG screening protocol (a hERG activation step) for visual purpose only
- [paper-states-occupancy.py](./paper-states-occupancy.py): Plot Beattie et al. 2018 sine wave, staircase-ramp protocol, current, and 4 states' occupancy
- [paper-protocols.py](./paper-protocols.py): Plot voltage clamp protocols
- [paper-fitting-and-validation-recordings.py](./paper-fitting-and-validation-recordings.py): Plot all selected cells data (both fitting and validation)
- [paper-fitting-and-validation-one-cell-zoom.py](./paper-fitting-and-validation-one-cell-zoom.py): Plot one cell model (fitting and validation) against data with zoom-in at selected peaks
- [paper-fitting-and-validation-selected-cells-zoom.py](./paper-fitting-and-validation-selected-cells-zoom.py): Plot all selected cells model (fitting and validation) against data with zoom-in at selected peaks
- [paper-rmsd-hist.py](paper-rmsd-hist.py): Plot RRMSE histograms with best, median, and 90 percentile; require output `//rmsd-matrix.txt` and `//rmsd-matrix-cells.txt` from [plot-rmsd-hist.py](./plot-rmsd-hist.py)
- [paper-swarmplot.py](./paper-swarmplot.py): Plot swarm plot for room temperature parameters (selectede cells) together with the parameters in Beattie et al. 2018.; and its violin-plot version [paper-violinplot.py](./paper-violinplot.py)
- [paper-pseudohbm-cov.py](./paper-pseudohbm-cov.py): Plot HBM from pseudo-MWG covariance/correlation matrix results (and with voltage error results)
- [paper-ek-hist.py](./paper-ek-hist.py): Plot an example of EK ramp fit and an estimated EK values histogram
- [paper-protocols-large.py](./paper-protocols-large.py): (Supplement) Plot voltage clamp protocols, large figure
- [paper-ramps.py](./paper-ramps.py): (Supplement) Plot an example of using the two ramps in the staircase protocol
- [paper-outliers-example.py](./paper-outliers-example.py): (Supplement) Plot manually removed wells example
- [paper-sweeps-comparison.py](./paper-sweeps-comparison.py): (Supplement) Plot fitted parameters from sweep 1 recording against fitted parameters from sweep 2 recordings
- [paper-qq-pp.py](./paper-qq-pp.py): (Supplement) Plot Q-Q plot and P-P plot for each parameter's marginal 1D, see <https://en.wikipedia.org/wiki/Normal_probability_plot> and <https://en.wikipedia.org/wiki/P-P_plot>


### Other plots (not used in publication)
- [plot-conductance.py](./plot-conductance.py): Plot conductance vs conductance/capacitance histograms, see if 'normalising' with Cm is helpful or not in CHO cells
- [plot-iv-protocols.py](./plot-iv-protocols.py): Plot short-activation and short-steady state inactivation protocols, mainly used for checking simulation, protocol, IV curves calculation working as expected
- [paper-fitting-and-validation-one-cell.py](./paper-fitting-and-validation-one-cell.py): (Not used) Plot one cell model (fitting and validation) against data
- [paper-fitting-and-validation-selected-cells.py](./paper-fitting-and-validation-selected-cells.py): (Not used) Plot all selected cells model (fitting and validation) against data
- [voltage-error-plot-cov-3.py](./voltage-error-plot-cov-3.py): Plot only the lower triangle for simplicity
- [plot-sweeps-comparison-ek.py](./plot-sweeps-comparison-ek.py): Plot the difference between sweep 1 and sweep 2 obtained parameters against the difference between the estimated reversal potential from the two sweeps; not much correlation, might mean we need better voltage error model

