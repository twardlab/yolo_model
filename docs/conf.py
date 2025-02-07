# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'yolo_model'
copyright = '2025, Andrew Bennecke, Daniel Tward'
author = 'Andrew Bennecke, Daniel Tward'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinxcontrib.htmlhelp'
]

# Configuration for LaTeX output
latex_documents = [
    ('index', 'yolo_help.tex', project, author, 'manual'),
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

import sys
sys.path.append('/home/runner/work/yolo_model/yolo_model/source')


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nature'
html_static_path = ['_static']
