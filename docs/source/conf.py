import datetime
import importlib
import os
import pathlib
import subprocess
import sys

import docutils.nodes
import docutils.parsers.rst.states
import tomllib

project = 'ReProspect'
author = 'Tomasetti, R and Arnst, M.'
copyright = f'{datetime.datetime.now(datetime.timezone.utc).year}, {author}'

DOCS_SOURCES_DIR = pathlib.Path(__file__).parent
DOCS_DIR         = DOCS_SOURCES_DIR.parent
PROJECT_DIR      = DOCS_DIR.parent

sys.path.append(str(PROJECT_DIR))

strategy_spec = importlib.util.spec_from_file_location('strategy', PROJECT_DIR / '.github' / 'workflows' / 'strategy.py')
strategy = importlib.util.module_from_spec(strategy_spec)
sys.modules['strategy'] = strategy
strategy_spec.loader.exec_module(strategy)

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
    'sphinx_copybutton',
    'sphinx_github_style',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.mermaid',
    'sphinxcontrib.tikz',
    'myst_nb',
]

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "navigation_depth": 8,
    "style_external_links": True,
}
html_static_path = ['_static']
html_logo = '_static/logo.svg'
html_last_updated_fmt = ''

# To the best of our knowledge, NVIDIA does not provide an object inventory.
extlinks = {
    'ncu_report': ('https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html#ncu_report.%s', '%s'),
}

intersphinx_mapping = {
    'cuda-bindings': ('https://nvidia.github.io/cuda-python/cuda-bindings/latest/', None),
    'cuda-core': ('https://nvidia.github.io/cuda-python/cuda-core/latest/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'packaging': ('https://packaging.pypa.io/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'python': ('https://docs.python.org/3', None),
    'rich': ('https://rich.readthedocs.io/en/stable/', None),
    'semantic_version': ('https://python-semanticversion.readthedocs.io/en/latest/', None),
}

autodoc_default_options = {
    'members': True,
    'special-members': '__str__,__init__,__enter__,__exit__',
    'show-inheritance': True,
    'undoc-members': True,
    'ignore-module-all': True,
}

autodoc_inherit_docstrings = False

apidoc_modules = [
    {
        'path': PROJECT_DIR / project.lower(),
        'destination': 'api',
        'max_depth': 4,
        'implicit_namespaces': True,
    },
]

apidoc_module_first = True
apidoc_separate_modules = True

bibtex_bibfiles = ['references.bib']

rst_prolog = f'''
.. _Kokkos: http://kokkos.org
.. _Kokkos Tools: https://github.com/kokkos/kokkos-tools
.. _Low-level Python Bindings for CUDA: https://nvidia.github.io/cuda-python/cuda-bindings/latest/
.. _CUDA binary utilities: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
.. _Nsight Compute: https://developer.nvidia.com/nsight-compute
.. _Nsight Systems: https://developer.nvidia.com/nsight-systems
.. _NVTX: https://github.com/NVIDIA/NVTX
.. _Google Benchmark: https://github.com/google/benchmark
.. _CMake: https://cmake.org
.. _std::mdspan: https://en.cppreference.com/w/cpp/container/mdspan.html
.. |kokkos_sha| replace:: {strategy.dependencies()['kokkos']['sha']}
'''

tikz_latex_preamble = r'\usepackage[dvipsnames]{xcolor}'

mermaid_d3_zoom = True

# Write the list of Docker images that we build into a file that can be included in the documentation.
matrix = strategy.build_matrix()
image_pairs = sorted(
    (entry["image"], entry["kokkos"])
    for entry in matrix
)
generated = DOCS_SOURCES_DIR / 'generated'
generated.mkdir(exist_ok=True)
lines = ['.. code-block:: shell', '']
lines += [f'   {image}\n   {kokkos}' for image, kokkos in image_pairs]
(generated / 'images.rst').write_text('\n'.join(lines) + '\n')

# Assert that the image names hardcoded in the "Running the tests" and "Running the examples" documentation still exist in the matrix
assert any(entry["image"] == "ghcr.io/uliegecsm/reprospect/cuda-gnu-14-nvidia-py3.13:13.1.0-devel-ubuntu24.04" for entry in matrix)
assert any(entry["kokkos"] == "ghcr.io/uliegecsm/reprospect/cuda-gnu-14-nvidia-py3.13-kokkos-5.1.0:13.1.0-devel-ubuntu24.04-blackwell120" for entry in matrix)

# 'unittest.TestCase' is implemented in 'unittest.test.TestCase' but is documented
# as 'unittest.TestCase', thus confusing 'intersphinx'.
import unittest

unittest.TestCase.__module__ = 'unittest'

import semantic_version

semantic_version.SimpleSpec.__module__ = 'semantic_version'
semantic_version.Version.__module__ = 'semantic_version'

import pandas

pandas.core.frame.DataFrame.__module__ = 'pandas'
pandas.core.series.Series.__module__ = 'pandas'

linkcode_url = 'https://github.com/uliegecsm/' + project.lower()

# Some references are broken, or the package does not provide an object inventory file.
# See also https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore_regex.
nitpick_ignore_regex = [
    ('py:class', r'blake3.blake3.*'),
    ('py:class', r'nvtx._lib.lib.*'),
    ('py:class', r'numpy.int64'),
    ('py:class', r'numpy._typing.*'),
    ('py:class', r'DTypeLike'),
    ('py:class', r'_regex.Match'),
    ('py:class', r'_regex.Pattern'),
    ('py:class', r'regex.Pattern'),
    ('py:class', r'elftools.*'),
    ('py:class', r'cmake_file_api.*'),
]

# Configuration for 'myst_nb', see also https://myst-nb.readthedocs.io/en/latest/configuration.html.
nb_merge_streams = True
nb_execution_in_temp = True

def get_last_commit(*, file: pathlib.Path, cwd: pathlib.Path) -> str:
    """
    Get the last commit hash that modified `file`.
    """
    cmd = ('git', 'log', '-n', '1', '--pretty=format:%H', '--', file)
    return subprocess.check_output(args=cmd, cwd=cwd, text=True).strip()

def lastcommit(name: str, rawtext: str, text: str, lineno: int, inliner: docutils.parsers.rst.states.Inliner, **kwargs) -> tuple[list[docutils.nodes.Node], list[docutils.nodes.system_message]]:
    """
    References:

    * https://www.sphinx-doc.org/en/master/development/tutorials/extending_syntax.html#writing-the-extension
    """
    commit_hash = get_last_commit(file=pathlib.Path(text), cwd=PROJECT_DIR)
    url = f'{linkcode_url}/commit/{commit_hash}'
    node = docutils.nodes.reference(
        rawsource=rawtext,
        text=project.lower() + '@' + commit_hash[:7],
        refuri=url,
        **kwargs,
    )
    return [node], []

def repofile(name: str, rawtext: str, text: str, lineno: int, inliner: docutils.parsers.rst.states.Inliner, **kwargs) -> tuple[list[docutils.nodes.Node], list[docutils.nodes.system_message]]:
    """
    Role linking a repository file to its content on GitHub, pinned to the file's last commit.
    """
    commit_hash = get_last_commit(file=pathlib.Path(text), cwd=PROJECT_DIR)
    url = f'{linkcode_url}/blob/{commit_hash}/{text}'
    node = docutils.nodes.reference(
        rawsource=rawtext,
        text=text,
        refuri=url,
        **kwargs,
    )
    return [node], []

def setup(app):
    app.add_role('lastcommit', lastcommit)
    app.add_role('repofile', repofile)
