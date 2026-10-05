from app import scoring


def test_best_respects_direction():
    assert scoring.best_of([12.5, 11.9, 13.0], "lower") == 11.9
    assert scoring.best_of([10, 14, 12], "higher") == 14
    assert scoring.best_of([None, None], "lower") is None


def test_change_direction_aware():
    d, imp = scoring.change(12.0, 11.0, "lower")
    assert d == -1.0 and imp is True
    d, imp = scoring.change(10, 8, "higher")
    assert imp is False
    assert scoring.change(5, 5, "lower")[1] is None


def test_histogram_counts_all():
    vals = [1, 2, 2, 3, 4, 5, 9]
    assert sum(b[2] for b in scoring.histogram(vals)) == len(vals)
    assert scoring.histogram([7, 7]) == [(7, 7, 2)]


def test_classify_without_rows_is_none():
    assert scoring.classify(10, []) is None
