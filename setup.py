from setuptools import setup, find_packages

setup(
    name="sim_eval",
    version="0.4",
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
    ],
    extras_require={
        "nequip": ["nequip"],
        "chgnet": ["chgnet"],
        "all": ["nequip", "chgnet"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 Licence",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6,<3.12',
)