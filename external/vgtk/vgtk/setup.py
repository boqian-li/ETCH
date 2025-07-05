from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# >>>>>>>>>>>>>>>> Notes <<<<<<<<<<<<<<<<<<#
"""
this is a custom
"""
# import epn_zpconv as cuda_zpconv
# import epn_gathering as gather
# import epn_grouping as cuda_nn
setup(
    name='se3-gconv',
    ext_modules=[
        CUDAExtension('epn_zpconv', [
            'cuda/zpconv_cuda.cpp',
            'cuda/zpconv_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']}),
        CUDAExtension('epn_gathering', [
            'cuda/gathering_cuda.cpp',
            'cuda/gathering_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']}),
        CUDAExtension('epn_grouping', [
            'cuda/grouping_cuda.cpp',
            'cuda/grouping_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})

    ],
    cmdclass={'build_ext': BuildExtension}
)
