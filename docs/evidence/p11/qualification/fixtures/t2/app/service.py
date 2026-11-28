from app.models import User


def adult_names(users):
    return [u.name for u in users if u.age > 18]
