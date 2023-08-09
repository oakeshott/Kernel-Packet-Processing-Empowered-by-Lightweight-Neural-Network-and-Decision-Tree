#!/usr/bin/bash

######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : run
# @created     : Wednesday Dec 07, 2022 12:22:08 JST
#
# @description : 
######################################################################

python learn_dt.py --dataroot flows.pickle --function train_dt --max_depth 5 --is-binary
python learn_dt.py --dataroot flows.pickle --function train_dt --max_depth 10 --is-binary
python learn_dt.py --dataroot flows.pickle --function train_dt --max_depth 5
python learn_dt.py --dataroot flows.pickle --function train_dt --max_depth 10
python learn_rf.py --dataroot flows.pickle --function train_dt --max_depth 5 --is-binary
python learn_rf.py --dataroot flows.pickle --function train_dt --max_depth 10 --is-binary
python learn_rf.py --dataroot flows.pickle --function train_dt --max_depth 5
python learn_rf.py --dataroot flows.pickle --function train_dt --max_depth 10

python learn_dt_float.py --dataroot flows.pickle --function train_dt --max_depth 5 --is-binary
python learn_dt_float.py --dataroot flows.pickle --function train_dt --max_depth 10 --is-binary
python learn_dt_float.py --dataroot flows.pickle --function train_dt --max_depth 5
python learn_dt_float.py --dataroot flows.pickle --function train_dt --max_depth 10
python learn_rf_float.py --dataroot flows.pickle --function train_dt --max_depth 5 --is-binary
python learn_rf_float.py --dataroot flows.pickle --function train_dt --max_depth 10 --is-binary
python learn_rf_float.py --dataroot flows.pickle --function train_dt --max_depth 5
python learn_rf_float.py --dataroot flows.pickle --function train_dt --max_depth 10

make
MODEL_PATH=runs/dt/binary-classification/fixed-maxdepth5
python test_dt.py --dataroot flows.pickle --function train_dt --model-path $MODEL_PATH --is-binary
MODEL_PATH=runs/dt/binary-classification/fixed-maxdepth10
python test_dt.py --dataroot flows.pickle --function train_dt --model-path $MODEL_PATH --is-binary
MODEL_PATH=runs/rf/binary-classification/fixed-maxdepth5-n_estimators10
python test_rf.py --dataroot flows.pickle --function train_dt --model-path $MODEL_PATH --is-binary
MODEL_PATH=runs/rf/binary-classification/fixed-maxdepth10-n_estimators10
python test_rf.py --dataroot flows.pickle --function train_dt --model-path $MODEL_PATH --is-binary
MODEL_PATH=runs/dt/multi-classification/fixed-maxdepth5
python test_dt.py --dataroot flows.pickle --function train_dt --model-path $MODEL_PATH
MODEL_PATH=runs/dt/multi-classification/fixed-maxdepth10
python test_dt.py --dataroot flows.pickle --function train_dt --model-path $MODEL_PATH
MODEL_PATH=runs/rf/multi-classification/fixed-maxdepth5-n_estimators10
python test_rf.py --dataroot flows.pickle --function train_dt --model-path $MODEL_PATH
MODEL_PATH=runs/rf/multi-classification/fixed-maxdepth10-n_estimators10
python test_rf.py --dataroot flows.pickle --function train_dt --model-path $MODEL_PATH
