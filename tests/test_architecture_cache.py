import json
import multiprocessing as mp

import pytest

from architecture_cache import ArchitectureCache, net_list_signature

NET_A = ['conv_3', 'pool', 'no_op']
NET_B = ['conv_5', 'pool', 'no_op']


@pytest.fixture
def cache(tmp_path):
    return ArchitectureCache(str(tmp_path / 'exp' / 'cache.json'))


def test_signature_is_stable_and_order_sensitive():
    assert net_list_signature(NET_A) == 'conv_3|pool|no_op'
    assert net_list_signature(NET_A) != net_list_signature(NET_A[::-1])


def test_init_creates_directory_and_empty_file(cache):
    assert open(cache.cache_path).read() == ''
    assert cache.find_cached_result(NET_A) is None


def test_init_does_not_clobber_existing_cache(tmp_path):
    path = str(tmp_path / 'cache.json')
    ArchitectureCache(path).register(NET_A, 90.0, 1.5, 200.0)
    reopened = ArchitectureCache(path)
    assert reopened.find_cached_result(NET_A)['fitness'] == 90.0


def test_miss_then_register_then_hit(cache):
    assert cache.find_cached_result(NET_A) is None
    cache.register(NET_A, fitness=91.5, params_m=1.2, inference_us=340.0)
    hit = cache.find_cached_result(NET_A)
    assert hit['fitness'] == 91.5
    assert hit['params_m'] == 1.2
    assert hit['inference_us'] == 340.0


def test_distinct_architectures_do_not_collide(cache):
    cache.register(NET_A, 80.0, 1.0, 1.0)
    assert cache.find_cached_result(NET_B) is None


def test_hit_count_increments_on_each_hit(cache):
    cache.register(NET_A, 80.0, 1.0, 1.0)
    counts = [cache.find_cached_result(NET_A)['hit_count'] for _ in range(3)]
    assert counts == [1, 2, 3]


def test_misses_do_not_touch_hit_counts(cache):
    cache.register(NET_A, 80.0, 1.0, 1.0)
    cache.find_cached_result(NET_B)
    assert json.load(open(cache.cache_path))['conv_3|pool|no_op']['hit_count'] == 0


def test_reregister_preserves_hit_count_but_updates_values(cache):
    cache.register(NET_A, 80.0, 1.0, 1.0)
    cache.find_cached_result(NET_A)
    cache.find_cached_result(NET_A)
    cache.register(NET_A, 85.0, 2.0, 3.0)
    entry = json.load(open(cache.cache_path))['conv_3|pool|no_op']
    assert entry == {'fitness': 85.0, 'params_m': 2.0, 'inference_us': 3.0, 'hit_count': 2}


def test_file_is_valid_json_after_writes(cache):
    for i in range(5):
        cache.register([f'op{i}'], float(i), 1.0, 1.0)
    assert len(json.load(open(cache.cache_path))) == 5


def _register_many(path, worker_id, n):
    cache = ArchitectureCache(path)
    for i in range(n):
        cache.register([f'w{worker_id}', f'i{i}'], float(i), 1.0, 1.0)


def test_concurrent_processes_lose_no_updates(tmp_path):
    """The flock-protected read-modify-write must serialize writers across processes."""
    path = str(tmp_path / 'cache.json')
    ArchitectureCache(path)
    workers, per_worker = 4, 15
    ctx = mp.get_context('fork')
    procs = [ctx.Process(target=_register_many, args=(path, w, per_worker)) for w in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert len(json.load(open(path))) == workers * per_worker
