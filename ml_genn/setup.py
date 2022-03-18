from setuptools import setup, find_packages

setup(
    name="ml_genn",
    version="2.0.0",
    packages=find_packages(),

    install_requires = [
        'pygenn>=4.7.0',
        'enum-compat',
        'tqdm']
)
