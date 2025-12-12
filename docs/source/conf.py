import datetime
import pathlib
import os
import sys
import subprocess
import tomllib

import docutils.nodes
import docutils.parsers.rst.states

project = 'ReProspect'
author = 'Tomasetti, R and Arnst, M.'
copyright = f'{datetime.datetime.now(datetime.timezone.utc).year}, {author}'

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
    'ncu_report': ('https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html#ncu_report.%s', '%s'),
}

intersphinx_mapping = {
    'cuda-bindings' : ('https://nvidia.github.io/cuda-python/cuda-bindings/latest/', None),
    'cuda-core' : ('https://nvidia.github.io/cuda-python/cuda-core/latest/', None),
    'matplotlib' : ('https://matplotlib.org/stable/', None),
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
        'path' : PROJECT_DIR / project.lower(),
        'destination' : 'api',
        'max_depth' : 4,
        'implicit_namespaces' : True,
    },
]

apidoc_module_first = True
apidoc_separate_modules = True

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

linkcode_url = 'https://github.com/uliegecsm/' + project.lower()

# Some references are broken, or the package does not provide an object inventory file.
# See also https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore_regex.
nitpick_ignore_regex = [
    ('py:class', r'blake3.blake3.*'),
    ('py:class', r'nvtx._lib.lib.*'),
    ('py:class', r'numpy.int64'),
    ('py:class', r'_regex.Match'),
    ('py:class', r'_regex.Pattern'),
    ('py:class', r'elftools.*'),
]

# Configuration for 'myst_nb', see also https://myst-nb.readthedocs.io/en/latest/configuration.html.
nb_merge_streams = True
nb_execution_in_temp = True

def get_last_commit(*, file : pathlib.Path, cwd : pathlib.Path) -> str:
    """
    Get the last commit hash that modified `file`.
    """
    cmd = ('git', 'log', '-n', '1', '--pretty=format:%H', '--', file)
    return subprocess.check_output(args = cmd, cwd = cwd, text = True).strip()

def lastcommit(name : str, rawtext : str, text : str, lineno : int, inliner : docutils.parsers.rst.states.Inliner, **kwargs) -> tuple[list[docutils.nodes.Node], list[docutils.nodes.system_message]]:
    """
    References:

    * https://www.sphinx-doc.org/en/master/development/tutorials/extending_syntax.html#writing-the-extension
    """
    commit_hash = get_last_commit(file = pathlib.Path(text), cwd = PROJECT_DIR)
    url = f'{linkcode_url}/commit/{commit_hash}'
    node = docutils.nodes.reference(
        rawsource = rawtext,
        text = project.lower() + '@' + commit_hash[:7],
        refuri = url,
        **kwargs,
    )
    return [node], []

def setup(app):
    app.add_role('lastcommit', lastcommit)
