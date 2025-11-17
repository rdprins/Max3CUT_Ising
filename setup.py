from setuptools import setup, find_packages
setup(
    name="max3cut_ising",
    version="0.0.0",
    packages=find_packages(),
    install_requires=["numpy","numba","pandas","scipy"],
)