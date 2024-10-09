#elastic_modulus
%%writefile elastic_modulus.py

import re
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

def elastic_modulus(content: str) -> List[np.ndarray] | None:
    pattern = r"Elastic Constant Matrix:\s*\(Units=GPa\)\s*\n-+\s*\n\s*Indices.*?\n-+\s*\n((?:\s*\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n){6})\s*-+"
    matches = re.findall(pattern, content, re.DOTALL)

    moduli_array = []
    for match in matches:
        matrix_lines = match.strip().split('\n')
        moduli = np.zeros((6, 6))
        for i, line in enumerate(matrix_lines):
            values = re.findall(r'[-\d.]+|\*+', line)[1:]
            for j, val in enumerate(values):
                try:
                    moduli[i, j] = float(val)
                except ValueError:
                    moduli[i, j] = 1e6
        moduli_array.append(moduli)
    if not moduli_array:
      return [np.full((6, 6), 1e6)]

    return moduli_array

%%writefile test_elastic_modulus.py
import pytest
import numpy as np
from elastic_modulus import elastic_modulus

def test_correct_data():
    content = """
  Elastic Constant Matrix: (Units=GPa)

-------------------------------------------------------------------------------
  Indices      1         2         3         4         5         6
-------------------------------------------------------------------------------
       1    103.9762   28.4985   16.8511   -4.6987   -0.0000   -0.0000
       2     28.4985  103.9762   16.8511    4.6987   -0.0000    0.0000
       3     16.8511   16.8511   37.3334    0.0000   -0.0000    0.0000
       4     -4.6987    4.6987    0.0000    8.6016   -0.0000   -0.0000
       5     -0.0000   -0.0000   -0.0000   -0.0000    8.6016   -4.6987
       6     -0.0000    0.0000    0.0000   -0.0000   -4.6987   37.7388
-------------------------------------------------------------------------------
    """
    result = elastic_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert result[0].shape == (6, 6)
    assert np.allclose(result[0], np.array([
        [103.9762,  28.4985,  16.8511,  -4.6987,  -0.0000,  -0.0000],
        [ 28.4985, 103.9762,  16.8511,   4.6987,  -0.0000,   0.0000],
        [ 16.8511,  16.8511,  37.3334,   0.0000,  -0.0000,   0.0000],
        [ -4.6987,   4.6987,   0.0000,   8.6016,  -0.0000,  -0.0000],
        [ -0.0000,  -0.0000,  -0.0000,  -0.0000,   8.6016,  -4.6987],
        [ -0.0000,   0.0000,   0.0000,  -0.0000,  -4.6987,  37.7388]
    ]))

def test_incorrect_data():
    content = """
  Elastic Constant Matrix:    (Units=GPa)

-------------------------------------------------------------------------------
Indices      1         2         3         4         5         6
-------------------------------------------------------------------------------
    1    ****      ****      ****       ****       ****       ****
    2    ****      ****      ****       ****       ****       ****
    3    ****      ****      ****       ****       ****       ****
    4    ****      ****      ****       ****       ****       ****
    5    ****      ****      ****       ****       ****       ****
    6    ****      ****      ****       ****       ****       ****
-------------------------------------------------------------------------------
    """
    result = elastic_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert result[0].shape == (6, 6)
    assert np.all(result[0] == 1e6)

def test_empty_data():
    content = """
    without any elastic constant matrix.
    """
    result = elastic_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert result[0].shape == (6, 6)
    assert np.all(result[0] == 1e6)

!python -m pytest test_elastic_modulus.py -v

#young_modulus
%%writefile young_modulus.py

import re
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

def young_modulus(content: str) ->  List[List[float]]:

    pattern = r"Youngs\s+Moduli\s*\(GPa\)\s*=\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)"
    matches = re.findall(pattern, content)

    def parse_value(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 1e6

    youngs_moduli = [[parse_value(val) for val in match] for match in matches]

    return youngs_moduli or [[1e6]]

%%writefile test_young_modulus.py

import pytest
import numpy as np
from young_modulus import young_modulus

def test_correct_data():
    content = """
    Youngs Moduli (GPa) =   103.9762   103.9762    37.3334
    """
    result = young_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert np.allclose(result[0], [103.9762, 103.9762, 37.3334])

def test_incorrect_data():
    content = """
    Youngs Moduli (GPa) =   ****   ****    ****
    """
    result = young_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert np.all(np.array(result[0]) == 1e6)

def test_empty_data():
    content = """
    without any Youngs Moduli data.
    """
    result = young_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0] == 1e6

!python -m pytest test_young_modulus.py -v

#bulk_modulus
%%writefile bulk_modulus.py

import re
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

def bulk_modulus(content: str) -> List[List[float]]:

    pattern = r"Bulk\s+Modulus\s*\(GPa\)\s*=\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)"
    matches = re.findall(pattern, content)

    def parse_value(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 1e6

    bulk_moduli = [[parse_value(val) for val in match] for match in matches]

    return bulk_moduli or [[1e6]]

%%writefile test_bulk_modulus.py

import pytest
import numpy as np
from bulk_modulus import bulk_modulus

def test_correct_data():
    content = """
    Bulk Modulus (GPa) =   50.2913    50.2913    50.2913
    """
    result = bulk_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert np.allclose(result[0], [50.2913, 50.2913, 50.2913])

def test_incorrect_data():
    content = """
    Bulk Modulus (GPa) =   ****   ****    ****
    """
    result = bulk_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert np.all(np.array(result[0]) == 1e6)

def test_empty_data():
    content = """
    without any Bulk Modulus data.
    """
    result = bulk_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0] == 1e6

!python -m pytest test_bulk_modulus.py -v

#shear_modulus
%%writefile shear_modulus.py

import re
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

def shear_modulus(content: str) -> List[List[float]]:

    pattern = r"Shear\s+Modulus\s*\(GPa\)\s*=\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)\s*([-\d.]+|\*+)"
    matches = re.findall(pattern, content)

    def parse_value(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 1e6

    shear_moduli = [[parse_value(val) for val in match] for match in matches]

    return shear_moduli or [[1e6]]

%%writefile test_shear_modulus.py

import pytest
import numpy as np
from shear_modulus import shear_modulus

def test_correct_data():
    content = """
    Mechanical properties :
    Shear Modulus (GPa) =   41.6375    41.6375    8.6016
    """
    result = shear_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert np.allclose(result[0], [41.6375, 41.6375, 8.6016])

def test_incorrect_data():
    content = """
    Mechanical properties :
    Shear Modulus (GPa) =   ****    ****    ****
    """
    result = shear_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 3
    assert np.all(np.array(result[0]) == 1e6)

def test_empty_data():
    content = """
    Shear Modulus data.
    """
    result = shear_modulus(content)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0] == 1e6

!python -m pytest test_shear_modulus.py -v

#phonon_gamma
%%writefile phonon_gamma.py

import re
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

def phonon_gamma(content: str, n: int) -> List[List[float]]:
    pattern = r"Frequencies \(cm-1\) \[NB: Negative implies an imaginary mode\]:\s*\n\s*((?:[-\d.\s]+\n)+)"
    frequency_blocks = re.findall(pattern, content, re.MULTILINE)

    all_frequencies = []

    for block in frequency_blocks:
        frequencies = re.findall(r'-?\d+\.?\d*', block)
        frequencies = [float(freq) for freq in frequencies]

        if len(frequencies) < n:
            frequencies.extend([-1000] * (n - len(frequencies)))

        frequencies = frequencies[:n]
        all_frequencies.append(frequencies)

    if not all_frequencies:
        return [[-1000] * n]

    return all_frequencies

%%writefile test_phonon_gamma.py
import pytest
import numpy as np
from phonon_gamma import phonon_gamma

def test_correct_data():
    content = """
    Frequencies (cm-1) [NB: Negative implies an imaginary mode]:
        -5.23    0.00    0.00   12.34   15.67   20.89
        25.12   30.45   35.78   40.01   45.23   50.56

    Some other text...

    Frequencies (cm-1) [NB: Negative implies an imaginary mode]:
        55.89   60.12   65.34   70.56   75.78   80.90
        85.12   90.34   95.56  100.78  105.90  110.12
    """
    result = phonon_gamma(content, 12)

    assert result is not None
    assert len(result) == 2
    assert len(result[0]) == 12
    assert len(result[1]) == 12
    assert np.allclose(result[0], [-5.23, 0.00, 0.00, 12.34, 15.67, 20.89, 25.12, 30.45, 35.78, 40.01, 45.23, 50.56])
    assert np.allclose(result[1], [55.89, 60.12, 65.34, 70.56, 75.78, 80.90, 85.12, 90.34, 95.56, 100.78, 105.90, 110.12])

def test_incomplete_data():
    content = """
    Frequencies (cm-1) [NB: Negative implies an imaginary mode]:
        -5.23    0.00    0.00   12.34   15.67
    """
    result = phonon_gamma(content, 12)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 12
    assert np.allclose(result[0][:5], [-5.23, 0.00, 0.00, 12.34, 15.67])
    assert np.all(np.array(result[0][5:]) == -1000)

def test_empty_data():
    content = """
    Frequencies data.
    """
    result = phonon_gamma(content, 12)

    assert result is not None
    assert len(result) == 1
    assert len(result[0]) == 12
    assert np.all(np.array(result[0]) == -1000)

!python -m pytest test_phonon_gamma.py -v

#phonon_kpoints
%%writefile phonon_kpoints.py

import re
import numpy as np
import numpy.typing as npt
from typing import List

def phonon_kpoints(content: str) -> List[List[float]]:
    pattern = r"K\s+point\s+\d+\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
    matches = re.findall(pattern, content)

    def parse_value(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return 1e6

    k_points = []
    for match in matches:
        try:
            point = [parse_value(val) for val in match]
            k_points.append(point)
        except Exception:
            k_points.append([1e6, 1e6, 1e6])

    return k_points if k_points else [[1e6, 1e6, 1e6]]

%%writefile test_phonon_kpoints.py
import pytest
import numpy as np
from phonon_kpoints import phonon_kpoints

def test_correct_data():
    content = """

--------------------------------------------------------------------------------
  K point      1 =   0.000000  0.000000  0.000000  Weight =    0.333
--------------------------------------------------------------------------------

    """
    result = phonon_kpoints(content)

    assert result is not None
    assert len(result) == 1
    assert result[0] == [0.0, 0.0, 0.0]

def test_incorrect_data():
    content = """

--------------------------------------------------------------------------------
  K point      5 =   ********  ********  ********  Weight =    *****
--------------------------------------------------------------------------------

    """
    result = phonon_kpoints(content)

    assert result is not None
    assert len(result) == 1
    assert result[0] == [1e6, 1e6, 1e6]

def test_empty_data():
    content = """
    k point data.
    """
    result = phonon_kpoints(content)

    assert result is not None
    assert len(result) == 1
    assert result[0] == [1e6, 1e6, 1e6]

!python -m pytest test_phonon_kpoints.py -v
