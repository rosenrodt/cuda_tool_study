Companion code for introductory usage of cuda profiler tools

A portion of code under `kernel_analysis` is adapted from https://github.com/angererc/nsight-gtc. Changes include 
* load jpg instead of ppm format
* iteratively widen image blocks to load into shared mem to demonstrate benefit of shared memory 
