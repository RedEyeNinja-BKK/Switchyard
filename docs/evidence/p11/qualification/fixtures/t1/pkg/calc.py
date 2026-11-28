def total(values):
    acc = 0
    for v in values:
        if isinstance(v, (int, float)):
            acc = acc + v
    return acc
