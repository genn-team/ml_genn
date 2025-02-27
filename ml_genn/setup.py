from setuptools import setup, find_packages

setup(name="ml_genn",
      version="2.3.0",
      packages=find_packages(),

      python_requires=">=3.7.0",
      install_requires=["pygenn>=5.1.0,<6.0.0",
                        "enum-compat", "deprecated",
                        "sympy>=1.9.0,<2.0.0",
                        "tqdm>=4.27.0", ])
