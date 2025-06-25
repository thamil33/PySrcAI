"""Install setuptools."""

import setuptools


def _remove_excluded(description: str) -> str:
  description, *sections = description.split('<!-- GITHUB -->')
  for section in sections:
    excluded, included = section.split('<!-- /GITHUB -->')
    del excluded
    description += included
  return description


with open('README.md', 'r', encoding='utf-8') as f:
  LONG_DESCRIPTION = _remove_excluded(f.read())


setuptools.setup(
    name='pysrcai',
    version='1.0.0',
    license='Apache 2.0',
    license_files=['LICENSE'],
    url='https://github.com/thamil33/PySrcAI',
    download_url='https://github.com/thamil33/PySrcAI/releases',
    author='PySrcAI Team',
    author_email='noreply@pysrcai.org',
    description=(
        'A framework for AI agents with geopolitical modeling capabilities.'
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords=(
        'multi-agent agent-based-simulation generative-agents python'
        ' machine-learning geopolitical-simulations ai-framework'
    ),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=setuptools.find_packages(),
    package_data={},
    # Explicitly include subpackage
    package_dir={"pysrcai": "pysrcai"},
    python_requires='>=3.11',
    install_requires=(
        'absl-py',
        'boto3',
        'google-cloud-aiplatform',
        'google-generativeai',
        'ipython',
        'langchain-community',
        'matplotlib',
        'mistralai',
        'numpy',
        'ollama',
        'openai>=1.3.0',
        'pandas',
        'python-dateutil',
        'reactivex',
        'retry',
        'termcolor',
        'together',
        'transformers',
        'typing-extensions',
        'jinja2',
    ),
    extras_require={
        # Used in development.
        'dev': [
            'build',
            'isort',
            'jupyter',
            'pipreqs',
            'pip-tools',
            'pyink',
            'pylint',
            'pytest-xdist',
            'pytype',
            'twine',
        ],
    },
)
