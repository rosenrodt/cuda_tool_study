# Code adapted from https://github.com/angererc/nsight-gtc
find_package(OpenCV REQUIRED)

add_executable(kernel_analysis demo_kernel_analysis.cu)
target_link_libraries(kernel_analysis PRIVATE
    opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui
)
target_compile_definitions(kernel_analysis PRIVATE
    -DOPTIMIZATION_STEP=0x20
)
target_compile_options(kernel_analysis PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>  # source-level profiling
)

# STEP 0x00: Default code
# STEP 0x1a: Change the block size to 32x2 (improve coalescing) -- It. 1, Eclipse Ed.
# STEP 0x1b: Change the block size to 8x16 (improve occupancy) -- It. 1, Visual Studio Ed.
# STEP 0x20: Change the block size to 32x4 (improve occupancy)
# STEP 0x30: Use launch_bounds to register pressure (improve occupancy) -- It. 2, Visual Studio Ed.
# STEP 0x40: Use shared memory (improve memory accesses)
# STEP 0x50: Use read-only path (reduce pressure on Load-store unit)
# STEP 0x5a: Optimized convolution filter 2D
# STEP 0x60: Implement a separable filter (reduce arithmetic intensity)
# STEP 0x70: Process two elements per thread (improve memory efficiency, increase ILP)
# STEP 0x80: Improve shared memory accesses (reduce bank conflicts)
# STEP 0x90: Use floats rather than ints (reduce pressure on arithmetic pipe)
# STEP 0x91: Use floats and math intrinsics in sobel_filter (compile with --use_fast_math)
