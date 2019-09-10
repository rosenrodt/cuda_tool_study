find_package(OpenCV REQUIRED)

add_executable(kernel_analysis demo_kernel_analysis.cu)
target_link_libraries(kernel_analysis PRIVATE
    opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui
)
target_compile_definitions(kernel_analysis PRIVATE
    -DOPTIMIZATION_STEP=0x00 
)