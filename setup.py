from setuptools import setup, find_packages

setup(
    name='DPT',
    version='3.0',
    description='Discrete Chirp Transform Software',
    author='Jaxson Mitchell',
    author_email='mitchj62@my.erau.edu',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'h5py'
    ],
)
