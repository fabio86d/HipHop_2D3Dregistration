from distutils.core import setup, Extension
nlopt_module = Extension('_nlopt',
                           sources=['nlopt-python.cpp'],
                           libraries=['libnlopt-0'],
                           )
import numpy
setup (name = 'nlopt',
       version = '2.4.2',
       author      = "Steven G. Johnson",
       description = """NLopt nonlinear-optimization library""",
       ext_modules = [nlopt_module],
       py_modules = ["nlopt"],
       include_dirs = ['.', numpy.get_include()],
       )
