from setuptools import setup, find_packages

setup(
    name="agentica_pyscrai",
    version="0.1.0",
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
