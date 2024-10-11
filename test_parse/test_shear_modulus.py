import pytest 
from ezff.parsers.gulp_parser import GulpParser

@pytest.mark.parametrize("content, expected_shape, expected_value", [
    ("""
    Mechanical properties :
    Shear Modulus (GPa) =   41.6375    41.6375    8.6016
    """, (1, 3), [41.6375, 41.6375, 8.6016]),
    ("""
    Mechanical properties :
    Shear Modulus (GPa) =   ****    ****    ****
    """, (1, 3), [1e6, 1e6, 1e6]),
    ("""
    Shear Modulus data.
    """, (1, 1), [1e6]),
])
def test_shear_modulus(parser, content, expected_shape, expected_value):
    shear_modulus_func = parser._GulpParser__extractors['shear_modulus']
    result = shear_modulus_func(content)

    assert result is not None
    assert len(result) == expected_shape[0]
    assert len(result[0]) == expected_shape[1]
    assert np.allclose(result[0], expected_value)
