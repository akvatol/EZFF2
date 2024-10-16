import pytest
from ezff.parsers.gulp_parser import GulpParser

@pytest.mark.parametrize("content, expected_shape, expected_value", [
    ("""
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
    """, (6, 6), 103.9762),
    ("""
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
    """, (6, 6), 1e6),
    ("without any elastic constant matrix.", (6, 6), 1e6),
])
def test_elastic_modulus(content, expected_shape, expected_value):
    elastic_modulus_func = parser._GulpParser__extractors['elastic_modulus']
    result = elastic_modulus_func(content)
    assert result is not None
    assert len(result) == 1
    assert result[0].shape == expected_shape
    assert result[0][0, 0] == expected_value
