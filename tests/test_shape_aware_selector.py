import pytest

from generation.shape_aware_selector import (
    choose_vector_launch_config,
    fixed_vector_launch_config,
    select_vector_launch_config,
)


def test_shape_aware_selector_uses_small_block_for_small_vectors():
    config = choose_vector_launch_config(1024)

    assert config.block_size == 128
    assert config.grid == (8,)
    assert "small vector" in config.reason


def test_shape_aware_selector_uses_balanced_block_for_medium_vectors():
    config = choose_vector_launch_config(1_048_576)

    assert config.block_size == 256
    assert config.grid == (4096,)
    assert "medium vector" in config.reason


def test_shape_aware_selector_uses_large_block_for_large_vectors():
    config = choose_vector_launch_config(2_000_000)

    assert config.block_size == 512
    assert config.grid == (3907,)
    assert "large vector" in config.reason


def test_shape_aware_selector_handles_empty_input():
    config = choose_vector_launch_config(0)

    assert config.block_size == 1
    assert config.grid == (0,)
    assert "empty input" in config.reason


def test_shape_aware_selector_rejects_negative_sizes():
    with pytest.raises(ValueError, match="num_elements"):
        choose_vector_launch_config(-1)


def test_fixed_launch_config_is_the_baseline_policy():
    config = fixed_vector_launch_config(1_048_576, block_size=1024)

    assert config.block_size == 1024
    assert config.grid == (1024,)
    assert "fixed baseline" in config.reason


def test_select_vector_launch_config_dispatches_by_policy():
    fixed = select_vector_launch_config(4096, policy="fixed", fixed_block_size=1024)
    shape_aware = select_vector_launch_config(4096, policy="shape-aware")

    assert fixed.block_size == 1024
    assert shape_aware.block_size == 128


def test_select_vector_launch_config_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unsupported launch policy"):
        select_vector_launch_config(4096, policy="unknown")
