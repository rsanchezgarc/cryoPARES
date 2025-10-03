# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from docutils.parsers.rst import Directive, directives
from docutils import nodes

# Add the project root to the path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CryoPARES'
copyright = '2025, Ruben Sanchez-Garcia'
author = 'Ruben Sanchez-Garcia'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',          # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',          # Add links to highlighted source code
    'sphinx.ext.intersphinx',       # Link to other project's documentation
    'sphinx.ext.autosummary',       # Generate autodoc summaries
]

# Try to import optional extensions
try:
    import sphinx_autodoc_typehints
    extensions.append('sphinx_autodoc_typehints')
except ImportError:
    pass

try:
    import myst_parser
    extensions.append('myst_parser')
except ImportError:
    pass

# Napoleon settings - support both Google and Sphinx style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'inherited-members': False  # Don't include inherited members from base classes
}

# Type hints
autodoc_typehints = 'description'
autodoc_type_aliases = {}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# MyST parser settings - support both .rst and .md
if 'myst_parser' in extensions:
    source_suffix = {
        '.rst': 'restructuredtext',
        '.md': 'markdown',
    }
    # Markdown extensions
    myst_enable_extensions = [
        'colon_fence',
        'deflist',
        'fieldlist',
    ]
else:
    source_suffix = '.rst'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Sidebar
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# Custom CSS
html_css_files = []

# Logo and favicon (add if you have them)
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# -- Custom handlers for PyTorch Lightning docstring compatibility -------------

def paramref_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Handle :paramref: role from PyTorch Lightning docstrings."""
    node = nodes.literal(rawtext, text)
    return [node], []

class TestCodeDirective(Directive):
    """Handle .. testcode:: directive from PyTorch Lightning docstrings."""
    has_content = True
    option_spec = {}

    def run(self):
        # Just create a literal block for test code
        code = '\n'.join(self.content)
        literal = nodes.literal_block(code, code)
        literal['language'] = 'python'
        return [literal]

def setup(app):
    """Register custom roles and directives."""
    app.add_role('paramref', paramref_role)
    app.add_directive('testcode', TestCodeDirective)
