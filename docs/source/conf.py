import datetime
import pathlib
import sys

project = 'ReProspect'
author = 'Tomasetti, R and Arnst, M.'
copyright = f'{datetime.datetime.now().year}, {author}'

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent

sys.path.append(str(PROJECT_DIR / 'python'))
from cuda_helpers import __version__
release = __version__

extensions = [
    'sphinx.ext.apidoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
]

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "style_external_links" : True,
}
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_last_updated_fmt = str()

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

apidoc_modules = [
    {
        'path' : PROJECT_DIR / 'python' / 'cuda_helpers',
        'destination' : 'api',
        'max_depth' : 4,
        'implicit_namespaces' : True,
    }
]
