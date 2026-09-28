"""Public-source test environments, independent of the private SWE verifier.

These are local development environments, not official scoring images. The runner
choices follow SWE-bench v3.0.9's public Python installation specifications; pins
below also keep old checkouts compatible with locally available Python builds.
Unknown versions fail closed instead of borrowing an unrelated environment.
"""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PublicTestRecipe:
    python: str
    packages: tuple[str, ...]
    runner: str
    smoke: str
    native_packages: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        fields = asdict(self)
        if not self.native_packages:
            fields.pop("native_packages")
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


RECIPES: dict[tuple[str, str], PublicTestRecipe] = {}


def _add(repo, versions, *, python="3.9", packages=(), runner="pytest", smoke, native_packages=()):
    for version in versions.split():
        RECIPES[repo, version] = PublicTestRecipe(
            python, (*(() if any(p.startswith("pip==") for p in packages) else ("pip==24.0",)),
                     *(() if any(p.startswith("setuptools==") for p in packages)
                                    else ("setuptools==68.0.0",)),
                     *(() if any(p.startswith("wheel==") for p in packages) else ("wheel==0.41.2",)), *packages),
            runner, smoke, native_packages,
        )


_PYTEST = ("pytest==7.4.0", "packaging==23.1")
_add("django/django", "1.11 3.0 3.1 3.2 4.0 4.1 4.2", runner="django",
     smoke="utils_tests.test_datastructures", packages=(
         "asgiref==3.7.2", "sqlparse==0.4.4", "pytz==2023.3", "tzdata==2023.3",
         "Jinja2==3.1.2", "Pillow==9.5.0", "docutils==0.19"))
_add("django/django", "5.0", python="3.11", runner="django",
     smoke="utils_tests.test_datastructures", packages=(
         "asgiref==3.7.2", "sqlparse==0.4.4", "tzdata==2023.3",
         "Jinja2==3.1.2", "Pillow==9.5.0", "docutils==0.19"))
_add("django/django", "1.11", python="3.6", runner="django", smoke="utils_tests.test_datastructures",
     packages=("pip==21.3.1", "setuptools==59.6.0", "wheel==0.37.1", "pytz==2023.3", "sqlparse==0.4.4"))
_add("sympy/sympy", "1.0 1.1 1.2 1.4 1.5 1.6 1.7 1.8 1.9 1.10 1.11 1.12 1.13 1.14",
     runner="sympy", smoke="sympy/core/tests/test_numbers.py",
     packages=("mpmath==1.3.0",))
_add("sympy/sympy", "1.0", runner="sympy", smoke="sympy/core/tests/test_cache.py",
     packages=("mpmath==1.3.0",))
_add("pallets/flask", "2.0 2.1 2.2 2.3", python="3.11", smoke="tests/test_basic.py",
     packages=(*_PYTEST, "Werkzeug==2.2.3", "Jinja2==3.1.2", "itsdangerous==2.1.2",
               "click==8.1.3", "MarkupSafe==2.1.3", "blinker==1.6.2", "asgiref==3.7.2"))
_add("psf/requests", "1.1 2.3 2.4", smoke="test_requests.py::RequestsTestCase::test_basic_building",
     packages=(*_PYTEST, "pytest-mock==3.11.1", "urllib3==1.26.18", "chardet==3.0.4",
               "idna==2.10", "certifi==2023.7.22", "mock==5.1.0"))
_add("pytest-dev/pytest", "5.1 5.4 6.0 6.2 6.3 7.2", smoke="testing/test_assertion.py::TestAssert_reprcompare",
     packages=("attrs==23.1.0", "iniconfig==2.0.0", "packaging==23.1", "pluggy==0.13.1",
               "py==1.11.0", "toml==0.10.2", "tomli==2.0.1", "more-itertools==10.1.0",
               "setuptools-scm==7.1.0", "hypothesis==6.82.6", "pygments==2.16.1", "atomicwrites==1.4.1"))
_add("pytest-dev/pytest", "4.6", smoke="testing/test_assertion.py::TestAssert_reprcompare",
     packages=("attrs==23.1.0", "packaging==23.1", "pluggy==0.13.1", "py==1.11.0",
               "more-itertools==10.1.0", "setuptools-scm==7.1.0", "hypothesis==6.82.6",
               "pygments==2.16.1", "atomicwrites==1.4.1", "six==1.16.0", "wcwidth==0.2.6",
               "importlib-metadata==1.7.0"))
_add("sphinx-doc/sphinx", "3.1 3.2 3.3 3.4 4.0 4.1 4.2 4.3 5.0", smoke="tests/test_util.py",
     packages=(*_PYTEST, "Jinja2==3.0.3", "MarkupSafe==2.0.1", "docutils==0.16",
               "sphinxcontrib-applehelp==1.0.4", "sphinxcontrib-devhelp==1.0.2",
               "sphinxcontrib-qthelp==1.0.3", "sphinxcontrib-htmlhelp==2.0.1",
               "sphinxcontrib-serializinghtml==1.1.5", "sphinxcontrib-jsmath==1.0.1",
               "alabaster==0.7.12", "Babel==2.12.1", "imagesize==1.4.1",
               "snowballstemmer==2.2.0", "requests==2.31.0", "Pygments==2.16.1",
               "pytest-xdist==3.3.1", "html5lib==1.1", "cython==0.29.36"))
_add("sphinx-doc/sphinx", "7.1", python="3.10", smoke="tests/test_util.py",
     packages=(*_PYTEST, "flit-core==3.9.0", "Jinja2==3.1.2", "MarkupSafe==2.1.3", "docutils==0.20.1",
               "sphinxcontrib-applehelp==1.0.4", "sphinxcontrib-devhelp==1.0.2",
               "sphinxcontrib-qthelp==1.0.3", "sphinxcontrib-htmlhelp==2.0.1",
               "sphinxcontrib-serializinghtml==1.1.5", "sphinxcontrib-jsmath==1.0.1",
               "alabaster==0.7.12", "Babel==2.12.1", "imagesize==1.4.1",
               "snowballstemmer==2.2.0", "requests==2.31.0", "Pygments==2.16.1",
               "pytest-xdist==3.3.1", "html5lib==1.1", "cython==0.29.36", "filelock==3.12.2"))
_add("pylint-dev/pylint", "2.10", smoke="tests/test_check_parallel.py",
     packages=("pytest==6.2.5", "packaging==23.1", "astroid==2.6.5", "isort==5.12.0",
               "mccabe==0.6.1", "toml==0.10.2", "pytest-benchmark==3.4.1"))
_add("pylint-dev/pylint", "2.14", smoke="tests/test_check_parallel.py",
     packages=("pytest==7.1.3", "packaging==23.1", "astroid==2.11.5", "dill==0.3.7",
               "isort==5.12.0", "mccabe==0.7.0", "platformdirs==3.10.0", "tomlkit==0.12.1",
               "tomli==2.0.1", "typing-extensions==4.7.1", "pytest-benchmark==3.4.1", "pytest-timeout==2.1.0"))
_add("pylint-dev/pylint", "2.15", smoke="tests/test_check_parallel.py",
     packages=(*_PYTEST, "astroid==2.12.13", "dill==0.3.7", "isort==5.12.0",
               "mccabe==0.7.0", "platformdirs==3.10.0", "tomlkit==0.12.1",
               "tomli==2.0.1", "typing-extensions==4.7.1", "pytest-benchmark==4.0.0"))
_add("pylint-dev/pylint", "3.0", python="3.11", smoke="tests/test_check_parallel.py",
     packages=(*_PYTEST, "astroid==3.0.0", "dill==0.3.7", "isort==5.12.0",
               "mccabe==0.7.0", "platformdirs==3.10.0", "tomlkit==0.12.1",
               "tomli==2.0.1", "typing-extensions==4.7.1", "pytest-benchmark==4.0.0"))
_add("pydata/xarray", "0.12 2022.03 2022.06 2022.09", python="3.10",
     smoke="xarray/tests/test_options.py",
     packages=(*_PYTEST, "numpy==1.23.5", "pandas==1.4.4", "scipy==1.10.1",
               "dask==2022.8.1", "cftime==1.6.2", "setuptools-scm==7.1.0"))
_add("matplotlib/matplotlib", "3.4 3.5 3.6 3.7", python="3.10",
     native_packages=("freetype=2.12.1", "qhull=2020.2"),
     smoke="lib/matplotlib/tests/test_cbook.py",
     packages=("pytest==7.1.3", "packaging==23.1", "numpy==1.23.5", "Pillow==9.5.0", "contourpy==1.1.0",
               "cycler==0.11.0", "fonttools==4.42.1", "kiwisolver==1.4.5",
               "pyparsing==3.0.9", "python-dateutil==2.8.2", "setuptools-scm==7.1.0",
               "certifi==2023.7.22", "pybind11==2.11.1", "setuptools-scm-git-archive==1.4"))
_add("scikit-learn/scikit-learn", "0.20 0.22", smoke="sklearn/utils/tests/test_multiclass.py",
     packages=(*_PYTEST, "numpy==1.19.5", "scipy==1.5.4", "cython==0.29.36",
               "joblib==1.3.2", "threadpoolctl==3.2.0", "setuptools==59.8.0", "six==1.16.0"))
_add("scikit-learn/scikit-learn", "0.21", python="3.6", smoke="sklearn/utils/tests/test_multiclass.py",
     packages=("pip==21.3.1", "setuptools==59.6.0", "wheel==0.37.1", "pytest==6.2.5",
               "packaging==21.3", "numpy==1.19.5", "scipy==1.5.4", "cython==0.29.36",
               "joblib==1.1.1", "threadpoolctl==3.1.0"))
_add("scikit-learn/scikit-learn", "1.3", python="3.10", smoke="sklearn/utils/tests/test_multiclass.py",
     packages=(*_PYTEST, "numpy==1.23.5", "scipy==1.10.1", "cython==0.29.36",
               "joblib==1.3.2", "threadpoolctl==3.2.0"))
_add("astropy/astropy", "5.0 5.1 5.2", python="3.10",
     smoke="astropy/units/tests/test_quantity.py",
     packages=("pytest==7.1.3", "numpy==1.23.5", "scipy==1.10.1", "cython==0.29.36",
               "pyerfa==2.0.0.3", "PyYAML==6.0.1", "extension-helpers==1.0.0",
               "setuptools-scm==7.1.0", "pytest-astropy==0.10.0", "hypothesis==6.82.6"))
_add("astropy/astropy", "1.3", python="3.6", smoke="astropy/units/tests/test_quantity.py",
     packages=("pip==21.3.1", "wheel==0.37.1", "numpy==1.16.0", "cython==0.29.36", "pytest==4.6.11",
               "astropy-helpers==3.2.2", "jinja2==3.0.3", "pytest-remotedata==0.3.3",
               "setuptools==44.1.1"))
_add("astropy/astropy", "3.1", python="3.6", smoke="astropy/units/tests/test_quantity.py",
     packages=("pip==21.3.1", "wheel==0.37.1", "numpy==1.16.0", "cython==0.29.36", "pytest==4.6.11",
               "astropy-helpers==3.2.2", "jinja2==3.0.3", "pytest-remotedata==0.3.3",
               "setuptools==44.1.1"))


def environment_key(repo: str, version: str) -> str:
    # Only known public identities can select a directory.
    if (repo, version) not in RECIPES:
        raise ValueError(f"unsupported public test environment: {repo}@{version}")
    return repo.replace("/", "__") + "--" + version
