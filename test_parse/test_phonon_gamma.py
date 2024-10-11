import pytest
from ezff.parsers.gulp_parser import GulpParser

@pytest.mark.parametrize("content, expected_shape, expected_value", [
    ("""
    Frequencies (cm-1) [NB: Negative implies an imaginary mode]:
        -5.23    0.00    0.00   12.34   15.67   20.89
        25.12   30.45   35.78   40.01   45.23   50.56

    other...

    Frequencies (cm-1) [NB: Negative implies an imaginary mode]:
        55.89   60.12   65.34   70.56   75.78   80.90
        85.12   90.34   95.56  100.78  105.90  110.12
    """, (2, 12), [[-5.23, 0.00, 0.00, 12.34, 15.67, 20.89, 25.12, 30.45, 35.78, 40.01, 45.23, 50.56],
                   [55.89, 60.12, 65.34, 70.56, 75.78, 80.90, 85.12, 90.34, 95.56, 100.78, 105.90, 110.12]]),
    ("""
    Frequencies (cm-1) [NB: Negative implies an imaginary mode]:
        -5.23    0.00    0.00   12.34   15.67
    """, (1, 12), [[-5.23, 0.00, 0.00, 12.34, 15.67] + [-1000] * 7]),
    ("""
    Frequencies data.
    """, (1, 12), [[-1000] * 12]),
])
def test_phonon_gamma(parser, content, expected_shape, expected_value):
    phonon_gamma_func = parser._GulpParser__extractors['phonon_gamma']
    result = phonon_gamma_func(content, 12)

    assert result is not None
    assert len(result) == expected_shape[0]
    for i in range(expected_shape[0]):
        assert len(result[i]) == expected_shape[1]
        assert np.allclose(result[i], expected_value[i])
