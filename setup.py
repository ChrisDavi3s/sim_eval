from setuptools import setup, find_packages

setup(
    name="sim_eval",
    version="0.1.0",
    description="A library for benchmarking simulation methods using various ASE calculators.",
    author="Chris Davies",
    author_email="",
    url="https://github.com/ChrisDavi3s/sim_eval",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "ase",
        "tqdm",
        "scipy",
        "nequip"
    ],
    extras_require={
        "nequip": ["nequip"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 Licence",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6,<3.12',
)