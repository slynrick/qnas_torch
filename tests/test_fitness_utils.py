import pytest

from cnn.fitness_utils import mofitness

# Limits: 2M params, 500 us inference.
KW = dict(T_p=2.0, T_t=500.0)


def test_within_limits_is_close_to_accuracy_scaled_to_100():
    f = mofitness(0.9, params=1.0, inference_time=250.0, **KW)
    # params_ratio=0.5, ratio**-0.01 ~ 1.007 -> slightly above 90
    assert 90.0 < f < 91.5


def test_accuracy_given_as_percentage_is_normalized():
    assert mofitness(90.0, 1.0, 250.0, **KW) == pytest.approx(mofitness(0.9, 1.0, 250.0, **KW))


def test_exactly_on_limits_is_pure_accuracy():
    assert mofitness(0.8, params=2.0, inference_time=500.0, **KW) == pytest.approx(80.0)


def test_exceeding_params_limit_is_penalized_heavily():
    ok = mofitness(0.9, 2.0, 500.0, **KW)
    over = mofitness(0.9, 4.0, 500.0, **KW)  # ratio 2 ** -1 = 0.5
    assert over == pytest.approx(ok * 0.5)


def test_exceeding_inference_limit_is_penalized_heavily():
    ok = mofitness(0.9, 2.0, 500.0, **KW)
    over = mofitness(0.9, 2.0, 1000.0, **KW)
    assert over == pytest.approx(ok * 0.5)


def test_higher_accuracy_wins_at_equal_cost():
    assert mofitness(0.95, 1.0, 100.0, **KW) > mofitness(0.90, 1.0, 100.0, **KW)


def test_smaller_model_wins_at_equal_accuracy():
    assert mofitness(0.9, 0.5, 100.0, **KW) > mofitness(0.9, 1.5, 100.0, **KW)


def test_loss_metric_is_monotonically_decreasing():
    low = mofitness(0.1, 1.0, 100.0, metric_type='loss', **KW)
    high = mofitness(2.0, 1.0, 100.0, metric_type='loss', **KW)
    assert low > high > 0


def test_zero_loss_maps_to_full_primary_fitness():
    assert mofitness(0.0, 2.0, 500.0, metric_type='loss', **KW) == pytest.approx(100.0)


def test_invalid_metric_type_raises():
    with pytest.raises(ValueError, match='metric_type'):
        mofitness(0.5, 1.0, 1.0, metric_type='f1', **KW)


def test_zero_params_yields_zero_fitness():
    assert mofitness(0.9, params=0.0, inference_time=100.0, **KW) == 0
