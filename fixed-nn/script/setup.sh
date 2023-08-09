#!/usr/bin/env sh

######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : setup
# @created     : Saturday Oct 22, 2022 00:06:52 JST
#
# @description : 
######################################################################


set -x
set -e

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
cd src && make
