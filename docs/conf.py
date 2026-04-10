# -- Sphinx Configuration for Fake News Detection Project ---------------------

project = 'Fake News Detection'
copyright = '2026, Hamdan'
author = 'Hamdan'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'review_outputs', 'project_review.py']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'Fake News Detection Documentation'

html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

# -- Autodoc -----------------------------------------------------------------

import os, sys
sys.path.insert(0, os.path.abspath('../src'))

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Napoleon (Google/NumPy docstrings) --------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
