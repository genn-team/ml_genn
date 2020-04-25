from setuptools import setup, find_packages

setup(
    name="tensor_genn",
    version="0.2.0",
    packages=find_packages(),

    install_requires = ['tensorflow>=2.0', 'pygenn>=0.2.1', 'enum-compat']
)
