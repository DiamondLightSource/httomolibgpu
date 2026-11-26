# -- General configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#!/usr/bin/env python
import os
import sys
from datetime import date
from unittest import mock

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("../.."))

# -- Mock imports -------------------------------------------------------------


# Mock imports instead of full environment in readthedocs
MOCK_MODULES = [
    "nvtx",
    "cupy",
    "cupyx",
    "numpy.polynomial",
    "tomobar",
    "tomobar.geometry",
    "tomobar.options",
    "tomobar.regularisers",
    "tomobar.reconstruction",
    "tomobar.reconstruction.direct",
    "tomobar.reconstruction.iterative",
    "tomobar.reconstruction.memory",
    "PIL",
    "skimage",
    "scipy",
    "scipy.fft",
    "scipy.ndimage",
    "pywt",
    "DeviceMemStack",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


class CustomMock(mock.Mock):
    def __repr__(self):
        return "<cp.ndarray>"


sys.modules["ccpi.filters.regularisersCuPy"] = CustomMock()
sys.modules["cupy"] = CustomMock()
sys.modules["pywt"] = CustomMock()
sys.modules["numpy"] = CustomMock()
sys.modules["numpy.polynomial"] = CustomMock()
sys.modules["cupyx.scipy.interpolate"] = CustomMock()
sys.modules["cupyx.scipy.ndimage"] = CustomMock()
sys.modules["cupyx.scipy.fft"] = CustomMock()
sys.modules["cupyx.scipy.fftpack"] = CustomMock()
sys.modules["scipy.fftpack"] = CustomMock()
sys.modules["tomobar"] = CustomMock()
sys.modules["tomobar.reconstruction.direct"] = CustomMock()
sys.modules["skimage.registration"] = CustomMock()
sys.modules["tomobar.reconstruction.memory"] = CustomMock()
# ------------------------------------------------------------------------------

project = "HTTomolibGPU"
author = "Imaging Team"
copyright = f"{date.today().year}, Diamond Light Source"

# Specify a base language to help assistive technology
language = "en"

# Save the commit hash, this is displayed in the page title
release = os.popen('git log -1 --format="%H"').read().strip()

# Set version as the latest tag in the current branch
version = os.popen("git describe --tags --abbrev=0").read().strip()

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    # Generates api files
    "sphinx.ext.autodoc",
    # Generates a short summary from docstring
    "sphinx.ext.autosummary",
    # Allows parsing of google style docstrings
    "sphinx.ext.napoleon",
    # Add links to highlighted source code
    "sphinx.ext.viewcode",
    # Allows a grid layout and dropdown boxes
    "sphinx_design",
    # copy to clipboard button
    "sphinx_copybutton",
    # use jupyter notebooks
    "sphinxcontrib.bibtex",
    # required for bibtex reference generation
    "nbsphinx",
    #'IPython.sphinxext.ipython_console_highlighting',
    "sphinx.ext.githubpages",
    # Generate .nojekyll file for git pages build
]

bibtex_bibfiles = ["bibliography/references.bib"]

autosummary_generate = True
numfig = True
# Document class attributes once rather than duplicating NumPy Attributes entries.
napoleon_use_ivar = True
# Each API page already identifies its module; keep function headings compact.
add_module_names = False
exclude_patterns = ["**/.ipynb_checkpoints", "**/.virtual_documents"]
# Build from saved results; documentation rendering must not require a GPU.
nbsphinx_execute = "never"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
templates_path = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "HTTomolibGPU Documentation"
html_copy_source = True
html_last_updated_fmt = ""
html_static_path = ["_static"]
html_css_files = ["css/general.css"]
html_codeblock_linenos_style = "table"

# Theme-aware SVG logos shared by the sidebar and documentation home link.
html_logo = "_static/logo-httomolibgpu-light.svg"

html_theme_options = {
    # Keep search in the sidebar; omit the redundant top navigation search.
    "navbar_persistent": [],
    "logo": {
        "image_light": "_static/logo-httomolibgpu-light.svg",
        "image_dark": "_static/logo-httomolibgpu-dark.svg",
        "alt_text": "HTTomolibGPU",
    },
    "show_toc_level": 1,
    "repository_url": "https://github.com/DiamondLightSource/httomolibgpu",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "show_navbar_depth": 1,
}

html_context = {
    "github_user": "DiamondLightSource",
    "github_repo": "httomolibgpu",
    "github_version": "main",
    "doc_path": "docs/source",
}


# Keep method guides reachable from every function's API entry. Check coverage
# from source rather than importing CUDA modules during documentation builds.
import ast
import json
from pathlib import Path


def _method_guide_links(app, what, name, obj, options, lines):
    if what == "function" and name in app.config.method_guides:
        guide = app.config.method_guides[name]
        lines[:0] = [
            f":doc:`Method guide: description, principles and usage </{guide}>`",
            "",
        ]


def _check_method_guides(app):
    from sphinx.errors import ExtensionError

    source = Path(app.srcdir)
    package = source.parent.parent / "httomolibgpu"
    public = set()
    for path in package.rglob("*.py"):
        relative = path.relative_to(package.parent)
        if any(
            part.startswith("_") and part != "__init__.py" for part in relative.parts
        ):
            continue
        module = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
        for node in ast.parse(path.read_text()).body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                public.add(f"{module}.{node.name}")
    guides = app.config.method_guides
    missing = public - guides.keys()
    stale = guides.keys() - public
    broken = [
        name for name, doc in guides.items() if not (source / (doc + ".rst")).is_file()
    ]
    if missing or stale or broken:
        raise ExtensionError(
            f"Method guide coverage: missing={sorted(missing)}, stale={sorted(stale)}, broken={broken}"
        )


def setup(app):
    guides = json.loads((Path(__file__).parent / "method_guides.json").read_text())
    app.add_config_value("method_guides", guides, "env")
    app.connect("builder-inited", _check_method_guides)
    app.connect("autodoc-process-docstring", _method_guide_links)


# Pin the renderer so Sphinx upgrades do not silently change equation layout.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
