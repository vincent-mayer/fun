from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='voxelize',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})