from arithmetic import divide


def test_integer_division_preserves_fraction():
    assert divide(3, 2) == 1.5
