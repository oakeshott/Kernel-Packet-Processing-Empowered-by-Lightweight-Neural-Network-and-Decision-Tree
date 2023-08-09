#!/usr/bin/env sh

######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : run
# @created     : Saturday Oct 22, 2022 00:09:30 JST
#
# @description : 
######################################################################
set -x
set -e
cd src
python train_mlp.py
python quantize_with_package.py
python create_convert_c_params.py
python test_mlp_c.py
