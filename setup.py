from setuptools import setup, find_packages

setup(
    name='dpt',
    version='1.0',
    description='Discrete Chirp Transform Software',
    author='Jaxson Mitchell',
    author_email='mitchj62@my.erau.edu',
    packages=[
        'dpt',
        'comp',
        'examples'
    ],
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'h5py'
    ],
)
