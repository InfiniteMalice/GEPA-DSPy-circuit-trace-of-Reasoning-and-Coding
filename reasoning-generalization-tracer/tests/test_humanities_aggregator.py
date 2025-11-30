import pytest

from rg_tracer.humanities.aggregator import HUMANITIES_AXES, HumanitiesProfile


def test_normalised_weights_rejects_boolean_values() -> None:
    axis = HUMANITIES_AXES[0]
    profile = HumanitiesProfile(name="demo", weights={axis: True})
    with pytest.raises(TypeError, match=axis):
        profile.normalised_weights()
