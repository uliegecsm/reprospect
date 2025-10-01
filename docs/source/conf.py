import datetime
import pathlib
import sys

project = 'ReProspect'
author = 'Tomasetti, R and Arnst, M.'
copyright = f'{datetime.datetime.now().year}, {author}'

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent

sys.path.append(str(PROJECT_DIR))
sys.path.append(str(PROJECT_DIR / 'python'))
from reprospect import __version__
release = __version__

extensions = [
    'sphinx.ext.apidoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinx_github_style',
]

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "style_external_links" : True,
}
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_last_updated_fmt = str()

intersphinx_mapping = {
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'python': ('https://docs.python.org/3', None),
}

autodoc_default_options = {
    'members' : True,
    'show-inheritance' : True,
    'undoc-members' : True,
}

apidoc_modules = [
    {
        'path' : PROJECT_DIR / 'python' / 'reprospect',
        'destination' : 'api',
        'max_depth' : 4,
        'implicit_namespaces' : True,
    }
]

bibtex_bibfiles = ['references.bib']

# 'unittest.TestCase' is implemented in 'unittest.test.TestCase' but is documented
# as 'unittest.TestCase', thus confusing 'intersphinx'.
import unittest
unittest.TestCase.__module__ = 'unittest'

linkcode_url = 'https://github.com/uliegecsm/reprospect'
