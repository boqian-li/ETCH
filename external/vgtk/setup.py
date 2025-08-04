#!/usr/bin/env python
"""Setup vgtk."""
from itertools import dropwhile
from setuptools import find_packages, setup
from os import path

from distutils.extension import Extension
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("vgtk", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def setup_package():
    setup(
        name="vgtk",
        version='1.0',
        packages=find_packages(exclude=["docs", "tests", "scripts"]),
    )

if __name__ == "__main__":
    setup_package()
