"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
import glob

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = None
with open(path.join(here, 'cryoPARES', '__init__.py'), encoding='utf-8') as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version

setup(
    name='cryoPARES',
    version=version,
    description='Cryo-EM Pose-Assignment for Related Experiments via Supervision',
    long_description=long_description,  # Optional
    url='https://github.com/rsanchezgarc/cryoPARES',  # Optional
    author='Ruben Sanchez-Garcia',  # Optional
    author_email='rsanchezgarc@faculty.ie.edu',  # Optional
    keywords='deep learning cryoem pose estimation',  # Optional
    packages=find_packages(),
    install_requires=[requirements],
    include_package_data=True,  # This line is important to read MANIFEST.in
    long_description_content_type="text/markdown",
)
