# Configuration file for the Sphinx documentation builder.

# Ensure Sphinx can find the package
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'GWADAMA'
copyright = '2025, Miquel Lluís Llorens Monteagudo'
author = 'Miquel Lluís Llorens Monteagudo'

from gwadama import __version__ as version
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    'numpydoc',            # Parses NumPy-style docstrings
    'sphinx.ext.autodoc',  # Auto-generates documentation from docstrings
    'sphinx.ext.viewcode', # Adds links to highlighted source code
    'sphinx.ext.mathjax',  # Renders math equations
    # 'sphinx.ext.autosummary',  # Generate summary tables
    'sphinx.ext.intersphinx',  # Allows references to external libraries
]

autosummary_generate = False
# autosummary_generate_overwrite = True  # Overwrite existing stub files
# autosummary_generate_output = os.path.join(os.path.abspath('.'), '_autosummary')
# autosummary_imported_members = True  # Include members imported from other modules
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "inherited-members": True,
    "show-inheritance": True,
}
exclude_patterns = []

suppress_warnings = ["autosummary"]

# Add explicit anchors to methods and class members
numpydoc_class_members_toctree = True  # Ensures class methods are linked in the "On this page" section
autodoc_default_flags = ['members', 'show-inheritance']

# To Enable Cross-Referencing with Sphinx (Intersphinx)
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}


# -- Options for HTML output -------------------------------------------------

html_static_path = ['_static']
templates_path = ['_templates']
html_theme = 'pydata_sphinx_theme'
html_css_files = [
    'custom.css',
]

# html_logo = "_static/gwadama-logo.svg"

html_theme_options = {
    # HEADER ------------------------------------------------------------------

    # "external_links": [
    #     {"name": "GitHub", "url": "https://github.com/MiquelLluis/GWADAMA"},
    # ],
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/MiquelLluis/GWADAMA",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
   ],


   # SECONDARY SIDEBAR (RIGHT) ------------------------------------------------

   "secondary_sidebar_items": ["page-toc"],
    "show_nav_level": 0,
    "show_toc_level": 2,
    "navigation_depth": 3,
}

