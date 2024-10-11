import pytest 
from ezff.parsers.gulp_parser import GulpParser

@pytest.mark.parametrize("content, expected_shape, expected_value", [
    ("""
    Bulk Modulus (GPa) =   50.2913    50.2913    50.2913
    """, (1, 3), [50.2913, 50.2913, 50.2913]),
    ("""
    Bulk Modulus (GPa) =   ****   ****    ****
    """, (1, 3), [1e6, 1e6, 1e6]),
    ("""
    without any Bulk Modulus data.
    """, (1, 1), [1e6]),
])
def test_bulk_modulus(parser, content, expected_shape, expected_value):
    bulk_modulus_func = parser._GulpParser__extractors['bulk_modulus']
    result = bulk_modulus_func(content)

    assert result is not None
    assert len(result) == expected_shape[0]
    assert len(result[0]) == expected_shape[1]
    assert np.allclose(result[0], expected_value)
