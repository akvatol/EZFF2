"""Interface to GULP, the General Utility Lattice Program."""

from pathlib import PurePath

import ase.io
from pyxtal import pyxtal

from .base_parser import BaseParser


class CifParser(BaseParser):
    """Instance of BaseParser with some extra extraction functions typically used when processing .out files."""

    def __init__(self, filepath: PurePath):
        super().__init__(filepath)
        self.__extractors = {
            "lattice": None,
            "atoms": None,
        }