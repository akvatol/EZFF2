from pathlib import PurePath
from typing import Callable  # noqa: UP035

import numpy as np
import numpy.typing as npt


class BaseParser:
    """Extendable parser for gulp out files."""

    def __init__(self, filepath: PurePath):
        self.filepath = filepath
        # TODO: Исправить типы
        self._data: dict[str, npt.NDArray[np.float64] | np.float64 | None] = {}
        self.__extractors: dict[str, Callable[[str], npt.NDArray[np.float64] | np.float64 | None]] = {}

    @property
    def extractors(self):  # noqa: D102
        return self.__extractors

    @extractors.setter
    def extractors(self, key: str, extractor: Callable[[str], npt.NDArray[np.float64] | np.float64 | None]) -> None:  # noqa: D102
        self.__extractors[key] = extractor

    @extractors.deleter
    def extractors(self, key: str) -> None:  # noqa: D102
        self.__extractors.pop(key, None)

    def parse(self) -> None:  # noqa: D102
        with open(self.filepath, 'r') as file:
            content = file.read()

        for key, extractor in self.extractors.items():
            self._data[key] = extractor(content)

    @property
    def data(self) -> dict:  # noqa: D102
        return self._data
