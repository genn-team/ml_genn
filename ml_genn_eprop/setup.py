from setuptools import setup, find_packages

setup(
    name="ml_genn_eprop",
    version="1.0.0",
    packages=find_packages(),

    python_requires=">=3.7.0",
    install_requires = [
        "ml_genn>=2.0.0"])
