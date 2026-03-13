#!/bin/bash

source install/setup.bash
conda activate bd_nav
export PYTHONPATH=$CONDA_PREFIX/lib/python3.10/site-packages:$PYTHONPATH
