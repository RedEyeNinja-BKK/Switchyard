from app.models import User
from app.service import adult_names


def test_boundary_is_inclusive():
    users = [User('a', 18), User('b', 17)]
    assert adult_names(users) == ['a']
