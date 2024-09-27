import pytest

from ezff.parsers.cif_parser import ParserCif


@pytest.mark.parametrize("file,expected", [
    ("/home/mpds_code/code/EZFF2/test_files/cif/NbS.cif", 194)
])
def test_group_number(file, expected):
    d = ParserCif(file)
    d.parse()
    assert d.data["group_number"] == expected