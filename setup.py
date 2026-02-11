from setuptools import setup, find_packages

setup(
    name="otk",
    version="0.1.0",
    description="ecDNA analysis tool based on deep learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "torch>=2.0",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
        "click>=8.1",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pyyaml>=6.0"
    ],
    entry_points={
        "console_scripts": [
            "otk=otk.cli:cli"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    python_requires=">=3.8"
)
