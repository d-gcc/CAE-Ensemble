# CAE-Ensemble

Code for the paper Time Series Outlier Detection with Diversity-Driven Convolutional Ensembles

How to run the model:
 * Execute [cae_ensemble.py](cae_ensemble.py) specifying the model parameters, for example:
 `python cae_ensemble.py --dataset 81 --diversity_factor 1 --beta_transfer 0.5 --rolling_size 16 --ensemble_members 20 --epochs 200`
 * The complete list of parameters is available at lines 963--1005 in [cae_ensemble.py](cae_ensemble.py). The model parameters are the ones detailed in 1.
 * The data sets number is related to the specification in lines 1064--1169.
 * The structure for the data input is defined in [data_provider.py](./utils/data_provider.py).
 * Results will be inserted in a database, calculations and connections are managed in [metrics_insert.py](./utils/metrics_insert.py).


Baselines:
 * Use the same structure as the CAE-Ensemble model, just using their specific parameters.

# Citation

If you use the code, please cite the following paper:

<pre>  
@article{pvldb/Ca22,
  author    = {David Campos and Tung Kieu and Chenjuan Guo and Feiteng Huang and Kai Zheng and 
               Bin Yang and Christian S. Jensen},
  title     = {{Unsupervised Time Series Outlier Detection with Diversity-Driven Convolutional 
               Ensembles}},
  journal   = {{PVLDB}},
  volume    = {15},
  number    = {3},
  pages     = {611--623},
  year      = {2022}
}
</pre> 
