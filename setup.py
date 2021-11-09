from distutils.core import setup, Extension

def main():
    setup(name="hebb_backend",
          version="1.0.0",
          description="C library functions for hebb",
          author="Clayton Seitz",
          author_email="cwseitz@uchicago.edu",
          ext_modules=[Extension("hebb_backend", ["hebb_backend.c"])])

if __name__ == "__main__":
    main()
