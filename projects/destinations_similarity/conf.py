# coding=utf-8
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
#sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, "/home")

# -- Project information -----------------------------------------------------
#from nightswatch.version import __version__

# The master toctree document.
master_doc = "index"

project = "Destination Similarity"
copyright = "2022, hurb.com"
author = "data.science@hurb.com"
release = "0.0.1"
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "env", ".flyte", "scripts"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = [".rst"]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# A boolean that decides whether codeauthor and sectionauthor directives produce any output in the
# built files.
show_authors = True

suppress_warnings = [
]

autodoc_mock_imports = ["flytekit", "faiss", "torch", "requests", "BeautifulSoup", "pandas", "streamlit", "numpy", "bs4",
                        "docker", "git", "typer", "sklearn", "streamlit", "argparse", "transformers", "swifter", "nltk",
                        "unidecode", "deep_translator"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "gray",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False
}

html_logo = "docs/images/vamoDalheLogo.jpeg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# ---------------------------------------------------
