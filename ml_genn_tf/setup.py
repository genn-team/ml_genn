from setuptools import setup, find_packages

setup(
    name="ml_genn_tf",
    version="2.3.0",
    packages=find_packages(),

    python_requires=">=3.7.0",
    install_requires = [
        "tensorflow<2.15.0",
        "ml_genn>=2.3.0,<3.0.0"])
