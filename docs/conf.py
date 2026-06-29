from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = 'mlx-lattice'
author = 'Z.Y. Lin'
copyright = f'{datetime.now(UTC).year}, {author}'
release = '0.2.0'
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'mlx-lattice'
html_theme_options = {
    'source_repository': 'https://github.com/utakotoba/mlx-lattice/',
    'source_branch': 'main',
    'source_directory': 'docs/',
}

autosummary_generate = True
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

nitpicky = False
suppress_warnings = ['autodoc.import_object']

os.environ.setdefault('PYTHONWARNINGS', 'default')
