from setuptools import setup, find_packages

setup(
    name="ml_genn",
    version="1.0.0",
    packages=find_packages(),

    install_requires = [
        'tensorflow>=2.0',
        'pygenn>=0.4.5',
        'enum-compat',
        'six',
        'tqdm']
)
