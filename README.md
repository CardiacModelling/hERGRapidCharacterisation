# Rapid characterisation of hERG channel kinetics

This repository contains all data, code and figures for the papers "*Rapid characterisation of hERG channel kinetics I: using an automated high-throughput system*" and "*Rapid characterisation of hERG channel kinetics II: temperature dependence*".


### Main results

[room-temperature-only](./room-temperature-only): Results of "*Rapid characterisation of hERG channel kinetics I: using an automated high-throughput system*".

[temperature-dependency](./temperature-dependency): Results of "*Rapid characterisation of hERG channel kinetics II: temperature dependence*".

[syn-room-temperature-only](./syn-room-temperature-only): Synthetic data studies supporting "*Rapid characterisation of hERG channel kinetics I: using an automated high-throughput system*".


### Code

The code requires Python 2.7 and two dependencies: [PINTS](https://github.com/pints-team/pints) and [Myokit](http://myokit.org).
[Matplotlib](https://pypi.org/project/matplotlib/) is required to regenerate the figures, and one of the figures also requires [Seaborn](https://pypi.org/project/seaborn/).


### Supporting files

[data](./data): Contains all `.csv` type data exported from automated QCs (see [qc](./qc)). Each protocol for each cell is saved as a separate file, currents are stored in the unit of [picoampere]. Time points for the current trace is saved as a separate file (to reduce duplicated information), and are in the unit of [second].

[data-sweep2](./data-sweep2): Same as [data](./data), but exporting second sweep of the recorded data, currently for staircaseramp protocol only.

[data-autoLC](./data-autoLC): Same as [data](./data), but using automated leak correction done by Nanion SynchoPatch384PE. Mainly for non-staircase ramp protocols, as we do not have an explicit leak estimation step in all of our protocols except staircase-ramp, and we hope their automated leak correction can do a better job if leak changes between protocols.

[data-raw](./data-raw): Same as [data](./data), but with raw data without any post-processing, contains `*-before.csv` and `*-after.csv`, referring to before and after E-4031 addition respectively; currently only contain `herg25oc1-staircaseramp` data.

[lib](./lib): Contains all external Python libraries (require manual installation, see [README](./lib/README.md)) and other modules/utility functions.

[mmt-model-files](./mmt-model-files): [Myokit](http://myokit.org/) model files, contains IKr model.

[protocol-time-series](./protocol-time-series): Contains protocols as time series, stored as `.csv` files, with time points (in [second]) and voltage (in [millivolt]).

[qc](./qc): Contains QC information for the selected cells in [data](./data), obtained using automated QC [hergqc.py](./lib/hergqc.py).

[supplement-info](./supplement-info): Contains extra information/tables for the Supporting Materials.


# Acknowledging this work

If you publish any work based on the contents of this repository please cite:

Lei, C. L., Clerx, M., Gavaghan, D. J., Polonchuk, L., Mirams, G. R., Wang, K.
(2019).
[Rapid characterisation of hERG channel kinetics I: using an automated high-throughput system](https://doi.org/10.1101/609727).
_bioRxiv_, 609727.

Lei, C. L., Clerx, M., Beattie, K. A., Melgari, D., Hancox, J., Gavaghan, D. J., Polonchuk, L., Wang, K., Mirams, G. R.
(2019).
[Rapid characterisation of hERG channel kinetics II: temperature dependence](https://doi.org/10.1101/609719).
_bioRxiv_, 609719.
