from setuptools import setup, find_packages
import pathlib
import re


def get_version():
    version_regex = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]")
    for line in pathlib.Path("spectral", "__init__.py").open("rt").readlines():
        mo = version_regex.match(line)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % ("spectral/__init__",))


def install_deps():
    """Reads requirements.txt and preprocess it
    to be feed into setuptools.

    This is the only possible way (we found)
    how requirements.txt can be reused in setup.py
    using dependencies from private github repositories.

    Links must be appendend by `-{StringWithAtLeastOneNumber}`
    or something like that, so e.g. `-9231` works as well as
    `1.1.0`. This is ignored by the setuptools, but has to be there.

    Warnings:
        to make pip respect the links, you have to use
        `--process-dependency-links` switch. So e.g.:
        `pip install --process-dependency-links {git-url}`

    Returns:
         list of packages and dependency links.
    """
    default = open("requirements.txt", "r").readlines()
    new_pkgs = []
    links = []
    for resource in default:
        if "git+" in resource:
            pkg = resource.split("#")[-1]
            links.append(resource.strip() + "-9876543210")
            new_pkgs.append(pkg.replace("egg=", "").rstrip())
        else:
            new_pkgs.append(resource.strip())
    return dict(install_requires=new_pkgs, dependency_links=links)


if __name__ == "__main__":
    setup(
        name="spectral",
        author="Max Kochurov",
        version=get_version(),
        packages=find_packages(),
        author_email="maxim.v.kochurov@gmail.com",
        **install_deps(),
    )
