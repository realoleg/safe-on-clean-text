from safe_on_clean_text.conformal.methods import jitter_seed
from safe_on_clean_text.degradations import make_stream, stream_seed


def test_stream_seeds_are_pinned_hashes():
    assert stream_seed(42, 5, "del") == 14038814787408674038
    assert stream_seed(42, 5, "typo") == 3006025793357987341
    assert stream_seed(42, 5, "mix") == 17568521401227026443
    assert jitter_seed(42, "resplit/0") == 10525357389781000074


def test_streams_do_not_depend_on_call_order():
    first = make_stream(42, 7, "del").random(5)
    make_stream(42, 8, "del").random(100)
    assert (make_stream(42, 7, "del").random(5) == first).all()
