add_executable(cuda_test
    hello_world.cu
)
target_link_libraries(cuda_test PRIVATE 
    ${NVTOOLS_EXT}
)
target_compile_options(cuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
