from app.service import adult_names


def render(users):
    return ', '.join(adult_names(users))
