# hERG paper II: temperature dependence

Fitting data across temperatures.


## Requirements

- myokit ([Myokit](http://myokit.org/))
- pints ([Pints](https://github.com/pints-team/pints))

Above libraries should be included/installed in [../lib/](../lib).


## Description

### Fitting
- [fit.py](./fit.py): Simple fitting for a cell; `Usage: python fit.py [int:file_id] [str:cell_id] --optional [N_repeats]`
- [fit.sh](./fit.sh): Bash script calling [fit.py](./fit.py) as a batch from [manualselection](./manualselection)
- [fit-multiple.py](./fit-multiple.py): Simple fitting for multiple selected cells, for testing mainly
- [fit-fromqc.py](./fit-fromqc.py): Fitting for all cells selected from [QC](../qc); prefer [fit.py](./fit.py)


### Manual selection
- [manualselection](./manualselection): Contains manual selected cells based on cells selected from [QC](../qc); `manualselected*` only based on staircaseramp protocol; `manualv2selected*` based on `manualselected*` and validation protocols.


### Hierarchical Bayesian Model (HBM)
- [pseudo2hbm.py](./pseudohbm.py): Run HBM using _simplified_ pseudo-MWG, requires individual cells' fitting results, and plot results


### Plots (Simple plots for testing)
- [plot-rmsd-hist.py](./plot-rmsd-hist.py): Plot fitting and validation (normalised) RMSD histograms for all selected cells (all temperatures)
- [plot-parameters-against-temperature-single-cell.py](./plot-parameters-against-temperature-single-cell.py): Quick plot for multiple selected cells, for testing mainly
- [plot-parameters-against-temperature.py](./plot-parameters-against-temperature.py): Quick plot for all fitted cells in [out](./out)
- `quick-plot-*`: Random quick testing plots


### Eyring/Q10
- [temperature\_models.py](./temperature_models.py): Contains functions and models for temperature dependence of ion channels, includes generalised Eyring relation and Q10 formulation.
- [eyring-q10-fit.py](./eyring-q10-fit.py): Fit Eyring relation and Q10 formulation to data, requires results from [pseudo2hbm.py](./pseudohbm.py); TODO, only able to plot in Eyring plot, need to do normal scale plot too.


### Others
- [generate-data-fancharts.py](./generate-data-fancharts.py): Generate fan charts lines for data to [./out](./out)
- [compare-normalisation-methods.py](./compare-normalisation-methods.py): Compare normalisation with 1. fitted conductance value and 2. extrapolation back to the step time at -120 mV step


### Paper figures
- [paper-rank-cells.py](./paper-rank-cells.py): Rank cells according to the hERG peak under hERG screening protocol (a hERG activation step)
- [paper-demo-eyring-q10.py](./paper-demo-eyring-q10.py): Plot demo of Generalised Eyring, Typical Eyring, and Q10 relationships
- [paper-fitting-and-validation-recordings.py](./paper-fitting-and-validation-recordings.py): Plot all selected cells data (both fitting and validation) for room and body temperatures
- [paper-fitting-and-validation-recordings-all.py](./paper-fitting-and-validation-recordings-all.py): Plot all selected cells data (both fitting and validation) for _all_ temperatures
- [paper-fitting-and-validation-selected-cells-zoom.py](./paper-fitting-and-validation-selected-cells-zoom.py): Plot all selected cells model (fitting and validation) against data with zoom-in at selected peaks; require argument `file-id`
- [paper-fitting-and-validation-selected-cells-zoom-run.sh](./paper-fitting-and-validation-selected-cells-zoom-run.sh): Run paper-fitting-and-validation-selected-cells-zoom.py](./paper-fitting-and-validation-selected-cells-zoom.py) for all temperatures
- [paper-rmsd-hist.py](./paper-rmsd-hist.py): Plot RRMSE histograms with best, median, and 90 percentile; require output `//rmsd-matrix.txt` and `//rmsd-matrix-cells.txt` from [plot-rmsd-hist.py](./plot-rmsd-hist.py); require argument `file-id`
- [paper-parameters.py](./paper-parameters.py): Plot all parameters as function of temperature, default in Eyring plot; optional argument `--normal` to plot it in 'linear scale'
- [paper-ss-tau.py](./paper-ss-tau.py): Plot steady states and time constants as function of voltage for all temperatures
- [paper-eyring-q10-fit.py](./paper-eyring-q10-fit.py): Plot all parameters as function of temperature with fits of Generalised Eyring and Q10 relationships, default in Eyring plot; optional argument `--normal` to plot it in 'linear scale'
- [paper-fitting-and-validation-eyring-q10.py](./paper-fitting-and-validation-eyring-q10.py): Plot predictions of HBM, Generalised Eyring, and Q10 against data in fan charts; require argument `protocol-id`; require results from [paper-eyring-q10-fit.py](./paper-eyring-q10-fit.py)
- [paper-fitting-and-validation-eyring-q10-v2.py](./paper-fitting-and-validation-eyring-q10-v2.py): [paper-fitting-and-validation-eyring-q10.py](./paper-fitting-and-validation-eyring-q10.py) version 2
- [paper-fitting-and-validation-eyring-q102-v2.py](./paper-fitting-and-validation-eyring-q102-v2.py): [paper-fitting-and-validation-eyring-q10-v2.py](./paper-fitting-and-validation-eyring-q10-v2.py) with Q10 model extrapolate 'down' from body temperature, instead of usual Q10 model which extrapolate 'up' from room temperature
- [paper-eyring-q10-ss-tau.py](./paper-eyring-q10-ss-tau.py): Same as [paper-parameters.py](./paper-parameters.py) with addition of Generalised Eyring and Q10 relationships results; require results from [paper-eyring-q10-fit.py](./paper-eyring-q10-fit.py)
- [paper-normalisation-demo.py](./paper-normalisation-demo.py): Plot demo of the 'maximum conductance value' estimation for normalisation purpose


### Reproducing literature
- `Vandenberg-2006-*`: Try reproducing Vandenberg et al. 2006 results
- [vandenberg-et-al-2006](./vandenberg-et-al-2006): Vandenberg et al. 2006 data
- `Mauerhofer-2016-*`: Try reproducing Mauerhofer et al. 2016 results
- [mauerhofer-et-al-2016](./mauerhofer-et-al-2016): Mauerhofer et al. 2016 data
- `Zhou-1998-*`: Try reproducing Zhou et al. 1998 results
- [zhou-et-al-1998](./zhou-et-al-1998): Zhou et al. 1998 data


## TODO

Try plotting fitting RMSD histogram for all temperature on the same plot -- if they are similar, then good, else if one is bigger or as temperature changes it changes, then might imply we are missing some state that temperature change allows its transition that we don't have in the model.

### Small tests
