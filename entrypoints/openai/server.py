"""Stable package entry point for the OpenAI HTTP adapter."""

from . import app as _impl
from entrypoints.cli.openai import main
import sys as _sys

_impl.main = main
_sys.modules[__name__] = _impl

if __name__ == "__main__":
    _impl.main()
