#!/usr/bin/bash

######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : tmp
# @created     : Thursday Dec 01, 2022 23:13:50 JST
#
# @description : 
######################################################################
BASEDIR=../saved_models/binary-classification/
models=`ls $BASEDIR`
for model in $models
do
  echo $model
  echo "TEST for Pytorch"
  ARR=(${model//x/ })
  python test_mlp.py --save_dir ${BASEDIR}/${model} --hidden_sizes ${ARR[0]} ${ARR[1]} --is-binary
  echo "TEST for C implement"
  python create_mlp_c_params.py --save_dir ${BASEDIR}/${model} --filename mlp_pktflw_quant.th --binary
  make
  python test_mlp_c.py --save_dir ${BASEDIR}/${model} --is-binary
done
BASEDIR=../saved_models/multi-classification/
models=`ls $BASEDIR`
for model in $models
do
  echo $model
  echo "TEST for Pytorch"
  ARR=(${model//x/ })
  python test_mlp.py --save_dir ${BASEDIR}/${model} --hidden_sizes ${ARR[0]} ${ARR[1]}
  echo "TEST for C implement"
  python create_mlp_c_params.py --save_dir ${BASEDIR}/${model} --filename mlp_pktflw_quant.th
  make
  python test_mlp_c.py --save_dir ${BASEDIR}/${model}
done
