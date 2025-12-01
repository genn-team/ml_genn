from os import path

from setuptools import setup

# Read version from txt file
abs_ml_genn_path = path.abspath(path.join(path.dirname(__file__), path.pardir))
with open(path.join(abs_ml_genn_path, "version.txt")) as version_file:
    version = version_file.read().strip()

setup(version=version)
