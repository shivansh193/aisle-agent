from aisle_agent.models import NemotronEngine


def engine(plain, listed):
    eng = NemotronEngine.__new__(NemotronEngine)
    eng._read_labels = lambda img, products: listed if products else plain
    return eng


def test_listed_reading_is_kept_only_where_the_plain_reading_saw_the_price():
    plain = [{"text": "Ar.. Lact..", "price": 60.0}, {"text": "", "price": 249.0}]
    listed = [{"text": "Arla Lactofree", "price": 60.0}, {"text": "Carrots", "price": 85.0}]   # 85: invented
    out = engine(plain, listed).read_labels(None, ["Arla Lactofree", "Carrots"])
    assert {"text": "Arla Lactofree", "price": 60.0} in out
    assert all(lab["price"] != 85.0 for lab in out)
    assert {"text": "", "price": 249.0} in out                      # read plainly, not tied to a product


def test_nothing_legible_means_no_second_call():
    eng = engine([], [{"text": "Carrots", "price": 85.0}])
    assert eng.read_labels(None, ["Carrots"]) == []
