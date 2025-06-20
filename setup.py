from setuptools import setup, find_packages

setup(
    name="pyscrai",
    version="0.5.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "numpy",
        "requests",
        "langchain",
        "sentence-transformers",
        "torch",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
        ]
    }
)
