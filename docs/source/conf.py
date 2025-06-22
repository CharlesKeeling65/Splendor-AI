# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# pylint: skip-file

from datetime import date

project = "Splendor-AI"
author = "Eyal Royee"
copyright = f"{date.today().year}, {author}"
release = "0.0.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # sphinx-builtin extensions
    "sphinx.ext.duration",       # Measure durations of Sphinx processing.
    "sphinx.ext.todo",           # Support for todo items & todolist in sphinx.
    "sphinx.ext.viewcode",       # Add links to highlighted source code.
    "sphinx.ext.autodoc",        # Include documentation from docstrings.
    "sphinx.ext.githubpages",    # Adds .nojekyll so everythings works in gh-pages.
    "sphinx.ext.graphviz",       # Uses graphviz for plotting graphs & visuals.

    # sphinx external extensions
    "sphinx_copybutton",         # Adds a copy button to each code block.
    "sphinx_rtd_dark_mode",      # Adds dark-mode theme to read-the-docs theme.
    "sphinxcontrib.plantuml",    # Uses plantuml for plotting graphs & visuals.
    "sphinxcontrib.mermaid",     # Uses mermaid for plotting graphs & visuals.
    "sphinxcontrib.shellcheck",  # Use shellcheck **inside** code blocks.
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_baseurl = "https://roeey777.github.io/Splendor-AI/"


# -- Custom configuration ----------------------------------------------------
# user starts in dark mode
default_dark_mode = True

# plantuml configurations
plantuml_output_format = "svg_img"
