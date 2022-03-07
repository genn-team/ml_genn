from setuptools import setup, find_packages

setup(
    name="ml_genn_tf",
    version="1.0.0",
    packages=find_packages(),

    install_requires = [
        'tensorflow>=2.0',
        'pygenn>=4.7.0',
        'enum-compat',
        'six',
        'tqdm']
)
