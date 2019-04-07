# hERGRapidCharacterisation

Rapid characterisation of hERG potassium channel kinetics using the staircase protocol.


### Main results

[room-temperature-only](./room-temperature-only): Results of *Rapid characterisation of hERG potassium channel kinetics I: using an automated high-throughput system*

[temperature-dependency](./temperature-dependency): Results of *Rapid characterisation of hERG potassium channel kinetics II: temperature dependence*

[syn-room-temperature-only](./syn-room-temperature-only): Synthetic data studies supporting *Rapid characterisation of hERG potassium channel kinetics I: using an automated high-throughput system*


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

*TBU*