# adversarial-recurrent-ids
Contact: Alexander Hartl, Maximilian Bachl. 

This repository contains the code and the figures for the [paper](https://ieeexplore.ieee.org/abstract/document/9179605) ([arXiv](https://arxiv.org/abs/1912.09855)) dealing with Explainability and Adversarial Robustness for RNNs. We perform our study in the context of Intrusion Detection Systems.

For the code of the *SparseIDS* paper please check out the *rl* branch. 

# Datasets
We use two datasets in the paper: CIC-IDS-2017 and UNSW-NB-15. The repository contains a preprocessed version of the datasets (refer to the paper for more information). 

```flows.pickle``` is for CIC-IDS-2017 while ```flows15.pickle``` is for UNSW-NB-15. Due to size constraints on GitHub they had to be split. Restore ```flows.pickle``` as follows:
* Concatenate the parts: ```cat flows.pickle.gz.part-* > flows.pickle.gz```
* Unzip them: ```gzip -d flows.pickle.gz```

Proceed analogously for ```flows15.pickle```. 

If you want to produce the preprocessed files yourself: 
* Follow the information on how to reproduce the preprocessed datasets in the [Datasets-preprocessing](https://github.com/CN-TU/Datasets-preprocessing) repository to generate labeled data for the `packet` feature vector.
* Run the ```parse.py``` script on the resulting ```.csv``` file to get the final ```.pickle``` file. 

# Usage examples
A list of command-line arguments is printed when calling `learn.py` without arguments. Here we provide a few examples of how `learn.py` can be called.
```bash
# Train a new model
./learn.py --dataroot flows.pickle
# Evaluate a model on the test dataset
./learn.py --dataroot flows.pickle --net runs/Oct26_00-03-50_gpu/lstm_module_1284.pth --function test
# Compute several feature importance metrics
./learn.py --dataroot flows.pickle --net runs/Oct26_00-03-50_gpu/lstm_module_1284.pth --function feature_importance
./learn.py --dataroot flows.pickle --net runs/Nov19_18-25-03_gpu/lstm_module_898.pth --function dropout_feature_importance
./learn.py --dataroot flows.pickle --net runs/Nov19_18-25-03_gpu/lstm_module_898.pth --function dropout_feature_correlation
./learn.py --dataroot flows.pickle --net runs/Oct26_00-03-50_gpu/lstm_module_1284.pth --adjustFeatImpDistribution --function mutinfo_feat_imp
./weight_feat_imp.py runs/Oct26_00-03-50_gpu/lstm_module_1284.pth
# Generate adversarial samples
./learn.py --dataroot flows.pickle --net runs/Oct26_00-03-50_gpu/lstm_module_1284.pth --function adv
# Perform adversarial training
./learn.py --dataroot flows.pickle --net runs/Oct26_00-03-50_gpu/lstm_module_1284.pth --advTraining
# Compute ARS
./learn.py --dataroot flows.pickle --net runs/Oct26_00-03-50_gpu/lstm_module_1284.pth --function adv_until_less_than_half
```

# Trained models
Models in the [runs](runs) folder have been trained with the following configurations:
* Oct26_00-03-50_gpu: CIC-IDS-2017
* Oct28_15-41-46_gpu: UNSW-NB-15
* Nov19_18-25-03_gpu: CIC-IDS-2017 with feature dropout
* Nov19_18-25-53_gpu: UNSW-NB-15 with feature dropout
* Nov20_18-27-31_gpu: CIC-IDS-2017 with adversarial training using L1 distance and CW with kappa=1
* Nov20_18-28-17_gpu: UNSW-NB-15 with adversarial training using L1 distance and CW with kappa=1
