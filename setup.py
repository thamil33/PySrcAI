from setuptools import setup, find_packages

setup(
    name="pyscrai",
    version="0.1.0",
    description="A package for scenario-based conversational AI components.",
    author="Tyler Hamilton",
    packages=find_packages(include=["pyscrai*", "concordia*"]),
    install_requires=[
        "numpy",
        "setuptools",
        "termcolor",
        "absl-py",
        "dotenv",
        "pandas",
        "requests",
        "ipython",
        "langchain-community",
        "matplotlib",
        "python-dateutil",
        "reactivex",
        "retry",
        "transformers",
        "typing-extensions",
        "jinja2",
    ],
)
