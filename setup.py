#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="rrgp",
    version="1.0.4",
    description="This project aims to train a model to recognise human activities (like walking, standing, or sitting) based on accelerometer and gyroscope data collected with a smartphone.",
    url="https://github.com/patrickaudriaz/mini-project",
    license="MIT",
    author="Geoffrey Raposo, Patrick Audriaz",
    author_email="geoffrey@raposo.ch, audriazp@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    entry_points={"console_scripts": ["rrgp-run = rrgp.run:main"]},
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
