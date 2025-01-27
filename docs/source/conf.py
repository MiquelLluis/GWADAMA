# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GWADAMA'
copyright = '2025, Miquel Lluís Llorens Monteagudo'
author = 'Miquel Lluís Llorens Monteagudo'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'numpydoc',            # Parses NumPy-style docstrings
    'sphinx.ext.autodoc',  # Auto-generates documentation from docstrings
    'sphinx.ext.viewcode', # Adds links to highlighted source code
    'sphinx.ext.mathjax',  # Renders math equations
    'sphinx.ext.autosummary', # Generate summary tables
]

autosummary_generate = True
autosummary_imported_members = True  # Include members imported from other modules

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "special-members": "__init__",
    "inherited-members": True,
    "show-inheritance": True,
}

# Optional: Numpydoc settings (tweak as needed)
# numpydoc_show_class_members = False  # Avoid showing all class members automatically
# numpydoc_class_members_toctree = False  # Avoid creating a toctree for class members

templates_path = ['_templates']
exclude_patterns = []

# Ensure Sphinx can find the 'clawdia' package
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']


# Add explicit anchors to methods and class members
numpydoc_class_members_toctree = True  # Ensures class methods are linked in the "On this page" section
autodoc_default_flags = ['members', 'show-inheritance']


html_theme_options = {
  "secondary_sidebar_items": ["page-toc", "sourcelink"],
  "show_toc_level": 2,
}

toc_object_entries_show_depth = 3
