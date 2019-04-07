# Libraries

For Python, it requires
- myokit ([Myokit](http://myokit.org/))
- pints ([Pints](https://github.com/pints-team/pints))

# Methods

It contains
- [model\_ikr](./model_ikr.py): Pints ForwardModel IKr model
- [parametertransform](./parametertransform.py): parameter transformation functions
- [priors](./priors.py): Pints LogPrior for IKr model
- [protocols](./protocols.py): protocols for [model\_ikr](./model_ikr.py)

- [plot\_hbm\_func.py](./plot_hbm_func.py): plotting functions, mainly for HBM
- [hbmdistribution](./hbmdistribution.py): theoretical posterior predictive distribution from HBM

Extra stuffs
- [hergqc](./hergqc.py): QC module that run the cell selection (all staircaseramp files in [data](../data) should pass this QC!)
