import os
import re
from setuptools import setup, find_packages


# Read the version from the package's __init__.py
def get_version():
    with open(os.path.join("gwadama", "__init__.py")) as f:
        match = re.search(r'__version__\s*=\s*["\'](.+?)["\']', f.read())
        if match:
            return match.group(1)
        raise RuntimeError("Version not found in __init__.py")
    

# Function to read dependencies from a file
def read_requirements(file):
    with open(file) as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]


setup(
    name="GWADAMA",
    version=get_version(),
    author="Miquel Lluís Llorens Monteagudo",
    author_email="miquel.llorens@uv.es",
    description="A Python package for gravitational wave dataset management.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MiquelLluis/GWADAMA",  # Update with your repository URL if applicable
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "optional": read_requirements("requirements-optional.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
