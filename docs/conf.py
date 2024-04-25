# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../ml_genn"))
sys.path.insert(0, os.path.abspath("../ml_genn_eprop"))
sys.path.insert(0, os.path.abspath("../ml_genn_tf"))


# -- Project information -----------------------------------------------------

project = "mlGeNN"
copyright = "2022, Jamie Knight, James Turner"
author = "Jamie Knight, James Turner"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "nbsphinx"]

napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock imports for readthedocs
autodoc_mock_imports = ["pygenn", "tensorflow"]

autodoc_typehints = "description"

autodoc_inherit_docstrings = True

# Combine __init__ documentation with class to remove need to duplicate in derived
autoclass_content = "both"

# Never actually run tutorial notebooks
nbsphinx_execute = "never"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

primary_domain = "py"

def skip_model_attributes(app, what, name, obj, skip, options):
    # If member is a value descriptor class attribute, skip it
    from ml_genn.utils.snippet import ConstantValueDescriptor
    from ml_genn.utils.value import ValueDescriptor
    if isinstance(obj, (ConstantValueDescriptor, ValueDescriptor)):
        return True
    else:
        return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_model_attributes)