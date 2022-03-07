from setuptools import setup, find_packages

setup(
    name="ml_genn_tf",
    version="1.0.0",
    packages=find_packages(),

    install_requires = [
        'tensorflow>=2.0',
        'ml_genn>=2.0.0']
)
