from pathlib import Path
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.rstrip() for line in f]

long_description = Path("README.md").read_text()

setup(
    name="gym_space",
    version="0.1",
    author="Kajetan Janiak",
    author_email="kajetan.janiak@gmail.com",
    description="RL environments with locomotion tasks in space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MIMUW-RL/gym-space",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
