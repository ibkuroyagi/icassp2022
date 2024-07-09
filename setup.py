#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="hearline_train",
    version="0.0.0",
    description="HEAR 2021 -- Line Model Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        # "tensorflow>=2.0",
        "heareval",
        "hearvalidator",
        "torchlibrosa",
        "efficientnet_pytorch",
    ],
)
