#!/bin/bash



#  PaddleBuild
pp="/backup/work/Paddle/build_gcc_Debug"

export PYTHONPATH="$pp/python"

cd $pp

#ctest -R test_fit_a_line  --verbose

#export FLAGS_init_allocated_mem=true
#export FLAGS_cudnn_deterministic=true 
#export FLAGS_cpu_deterministic=true

#export FLAGS_use_mkldnn=true

#export T_Y=true

export DNNL_VERBOSE=1 
#export T_X=true
#export T_BYY=true

cd python/paddle/fluid/tests/book/
/home/pablo/miniconda3/envs/paddle/bin/python3.8 test_fit_a_line.py
