from distutils.core import setup, Extension
import numpy

def main():

    setup(name="hebb_backend",
          version="1.0.0",
          description="C library functions for hebb",
          author="Clayton Seitz",
          author_email="cwseitz@uchicago.edu",
          ext_modules=[Extension("hebb_backend", ["hebb_backend.c"],
                       include_dirs = [numpy.get_include(), '/usr/include/gsl'],
                       library_dirs = ['/usr/lib/x86_64-linux-gnu'],
                       libraries=['m', 'gsl', 'gslcblas'])])


if __name__ == "__main__":
    main()
