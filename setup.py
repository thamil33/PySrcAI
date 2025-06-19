from setuptools import setup, find_packages

setup(
    name="pyscrai",
    version="0.1.0",
    description="A modular system utilizing concordia as a basis.",
    author="Tyler Hamilton",
    packages=find_packages(include=["pyscrai*", "concordia*"]),
    install_requires=[],
)
