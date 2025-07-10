"""Setup script for PySrcAI.

This module provides the necessary setup information to install PySrcAI
as a Python package.
"""

from setuptools import setup, find_packages

setup(
    name="pysrcai",
    version="0.1.0",
    description="A modular AI agent simulation framework",
    author="PySrcAI Team",
    author_email="info@pysrcai.org",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "numpy>=1.24.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0"
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
