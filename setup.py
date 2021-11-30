import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="zero-shot-image-to-text",
    py_modules=["zero-shot-image-to-text"],
    version="1.0",
    description="",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True
)