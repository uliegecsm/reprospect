import datetime
import pathlib
import os
import sys
import tomllib

project = 'ReProspect'
author = 'Tomasetti, R and Arnst, M.'
copyright = f'{datetime.datetime.now().year}, {author}'

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent

sys.path.append(str(PROJECT_DIR))

# Allow subprocesses launched by Sphinx to find ReProspect.
os.environ['PYTHONPATH'] = str(PROJECT_DIR) + os.path.pathsep + os.environ.get('PYTHONPATH', '')

with (PROJECT_DIR / 'pyproject.toml').open('rb') as f:
    release = tomllib.load(f)['project']['version']

extensions = [
    'sphinx.ext.apidoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx_github_style',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.mermaid',
    'sphinxcontrib.tikz',
    'sphinxemoji.sphinxemoji',
    'myst_nb',
]

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "style_external_links" : True,
}
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_last_updated_fmt = str()

# To the best of our knowledge, NVIDIA does not provide an object inventory.
extlinks = {
    'ncu_report': ('https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html#ncu_report.%s', '%s')
}

intersphinx_mapping = {
    'cuda-bindings' : ('https://nvidia.github.io/cuda-python/cuda-bindings/latest/', None),
    'cuda-core' : ('https://nvidia.github.io/cuda-python/cuda-core/latest/', None),
    'numpy' : ('https://numpy.org/doc/stable/', None),
    'packaging' : ('https://packaging.pypa.io/en/stable/', None),
    'pandas' : ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'python' : ('https://docs.python.org/3', None),
    'rich' : ('https://rich.readthedocs.io/en/stable/', None),
    'semantic_version' : ('https://python-semanticversion.readthedocs.io/en/latest/', None),
}

autodoc_default_options = {
    'members' : True,
    'special-members' : '__str__,__init__,__enter__,__exit__',
    'show-inheritance' : True,
    'undoc-members' : True,
}

apidoc_modules = [
    {
        'path' : PROJECT_DIR / 'reprospect',
        'destination' : 'api',
        'max_depth' : 4,
        'implicit_namespaces' : True,
    }
]

bibtex_bibfiles = ['references.bib']

rst_prolog = '''
.. _Kokkos: http://kokkos.org
.. _Low-level Python Bindings for CUDA: https://nvidia.github.io/cuda-python/cuda-bindings/latest/
.. _CUDA binary utilities: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
.. _Nsight Compute: https://developer.nvidia.com/nsight-compute
.. _Nsight Systems: https://developer.nvidia.com/nsight-systems
.. _CMake: https://cmake.org
'''

tikz_latex_preamble = r'\usepackage[dvipsnames]{xcolor}'

# 'unittest.TestCase' is implemented in 'unittest.test.TestCase' but is documented
# as 'unittest.TestCase', thus confusing 'intersphinx'.
import unittest
unittest.TestCase.__module__ = 'unittest'

import semantic_version
semantic_version.SimpleSpec.__module__ = 'semantic_version'
semantic_version.Version.__module__ = 'semantic_version'

linkcode_url = 'https://github.com/uliegecsm/reprospect'

# Some references are broken, or the package does not provide an object inventory file.
# See also https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore_regex.
nitpick_ignore_regex = [
    ('py:class', r'blake3.blake3.*'),
    ('py:class', r'nvtx._lib.lib.*'),
    ('py:class', r'numpy.int64'),
    ('py:class', r'_regex.Match'),
    ('py:class', r'_regex.Pattern'),
    ('py:class', r'NestedProfilingResults'),
    ('py:class', r'elftools.*'),
]

# Configuration for 'myst_nb', see also https://myst-nb.readthedocs.io/en/latest/configuration.html.
nb_merge_streams = True
nb_execution_in_temp = True
