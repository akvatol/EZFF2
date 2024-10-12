import pytest
from ezff.parsers.gulp_parser import GulpParser

@pytest.mark.parametrize("content, expected_shape, expected_value", [
    ("""
    Youngs Moduli (GPa) =   103.9762   103.9762    37.3334
    """, (1, 3), [103.9762, 103.9762, 37.3334]),
    ("""
    Youngs Moduli (GPa) =   ****   ****    ****
    """, (1, 3), [1e6, 1e6, 1e6]),
    ("""
    without any Youngs Moduli data.
    """, (1, 1), [1e6]),
])
def test_young_modulus(parser, content, expected_shape, expected_value):
    young_modulus_func = parser._GulpParser__extractors['young_modulus']
    result = young_modulus_func(content)

    assert result is not None
    assert len(result) == expected_shape[0]
    assert len(result[0]) == expected_shape[1]
    assert np.allclose(result[0], expected_value)
