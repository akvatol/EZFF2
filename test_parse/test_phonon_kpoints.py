import pytest
from ezff.parsers.gulp_parser import GulpParser

@pytest.mark.parametrize("content, expected_shape, expected_value", [
    ("""

--------------------------------------------------------------------------------
  K point      1 =   0.000000  0.000000  0.000000  Weight =    0.333
--------------------------------------------------------------------------------

    """, (1, 3), [[0.0, 0.0, 0.0]]),
    ("""

--------------------------------------------------------------------------------
  K point      5 =   ********  ********  ********  Weight =    *****
--------------------------------------------------------------------------------

    """, (1, 3), [[1e6, 1e6, 1e6]]),
    ("""
    k point data.
    """, (1, 3), [[1e6, 1e6, 1e6]]),
])
def test_phonon_kpoints(parser, content, expected_shape, expected_value):
    phonon_kpoints_func = parser._GulpParser__extractors['phonon_kpoints']
    result = phonon_kpoints_func(content)

    assert result is not None
    assert len(result) == expected_shape[0]
    assert len(result[0]) == expected_shape[1]
    assert np.allclose(result[0], expected_value[0])
