add_executable(hello_world
    hello_world.cu
)
target_link_libraries(hello_world PRIVATE 
    ${NVTOOLS_EXT} CUDA::nvToolsExt
)
target_compile_options(hello_world PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
