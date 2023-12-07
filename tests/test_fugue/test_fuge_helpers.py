def _compare_report(expected, actual, truncate=False):
    if truncate:
        expected = expected.split("Sample Rows", 1)[0]
        actual = actual.split("Sample Rows", 1)[0]
    assert expected == actual
