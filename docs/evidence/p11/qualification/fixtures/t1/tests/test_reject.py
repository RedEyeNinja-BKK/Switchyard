import pytest


from pkg.calc import total


def test_rejects_text():
    with pytest.raises(TypeError):
        total(['a'])
