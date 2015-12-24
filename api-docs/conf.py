#!/usr/bin/env python2

import sys
import os

from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('..'))


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
            return Mock()

MOCK_MODULES = ['argparse', 'numpy', 'pandas', 'cv2']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

project = 'OpenFace'
copyright = '2015, Carnegie Mellon University'
author = 'Carnegie Mellon University'

version = '0.1.1'
release = '0.1.1'

language = None

exclude_patterns = ['_build']

pygments_style = 'sphinx'

todo_include_todos = True


# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'OpenFacedoc'

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '12pt',
}

latex_documents = [
    (master_doc, 'OpenFace.tex', 'OpenFace Documentation',
     'Carnegie Mellon University', 'manual'),
]

man_pages = [
    (master_doc, 'openface', 'OpenFace Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'OpenFace', 'OpenFace Documentation',
     author, 'OpenFace', 'One line description of project.',
     'Miscellaneous'),
]
