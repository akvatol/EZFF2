"""Interface to GULP, the General Utility Lattice Program."""

from pathlib import PurePath

from .base_parser import BaseParser


class GulpParser(BaseParser):
    """Instance of BaseParser with some extra extraction functions typically used when processing .out files."""

    def __init__(self, filepath: PurePath):
        super().__init__(filepath)
        self.__extractors = {
            'energy': None,

            'atoms': None,
            'cell': None,

            'bulk_modulus': None,
            'young_modulus': None,
            'elastic_modulus': None,
            'shear_modulus': None,

            'phonon_disp': None,
            'phonon_gamma': None,
            'phonon_kpoints': None,

            'final_sum': None,
        }