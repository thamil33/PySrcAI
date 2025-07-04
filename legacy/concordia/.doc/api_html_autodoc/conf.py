# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# set of options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Ensure Sphinx can import the concordia package from two levels up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# -- Project information -----------------------------------------------------

project = 'Concordia'
copyright = '2025, PySrcAI Team'
author = 'PySrcAI Team'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
}

# Paths for autodoc
autodoc_mock_imports = []

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

# -- Napoleon settings (for Google/NumPy style docstrings) -------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
