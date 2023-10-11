from setuptools import setup, find_packages

setup(name="ml_genn",
      version="2.1.0",
      packages=find_packages(),

      python_requires=">=3.7.0",
      install_requires=["pygenn>=4.9.0,<5.0.0",
                        "enum-compat",
                        "tqdm>=4.27.0", "deprecated"])
