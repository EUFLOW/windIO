"""Functional coverage for mixed turbine types and per-turbine wind resources.

The windIO standard supports assigning different turbine types (and therefore
different hub heights) to different positions in a wind farm, and supports
wind-resource data dimensioned per turbine. These tests assert that those
structures parse and round-trip through windIO's public API, rather than only
being validated implicitly via the example-validation sweep.
"""

from pathlib import Path

import windIO


def _plant_examples_dir():
    return Path(windIO.plant_ex.__file__).parent


def test_mixed_turbine_types_wind_farm():
    """A wind farm assigns >1 turbine type by per-position index, and the
    referenced types have distinct hub heights (mixed hub heights)."""
    farm_yaml = _plant_examples_dir() / "plant_wind_farm" / "multiple_types.yaml"

    # Mixed types are a first-class windIO feature -> must validate.
    windIO.validate(input=farm_yaml, schema_type="plant/wind_farm")

    farm = windIO.load_yaml(farm_yaml)
    layout = farm["layouts"][0]
    type_idx = layout["turbine_types"]
    n_positions = len(layout["coordinates"]["x"])

    # A per-position type index referencing a multi-entry turbine_types map.
    assert len(farm["turbine_types"]) >= 2
    assert len(type_idx) == n_positions
    assert set(type_idx) == set(farm["turbine_types"].keys())
    # The example genuinely uses both types.
    assert len(set(type_idx)) >= 2

    # The assigned types have distinct hub heights -> mixed hub heights.
    hub_heights = {k: t["hub_height"] for k, t in farm["turbine_types"].items()}
    assert len(set(hub_heights.values())) >= 2


def test_per_turbine_wind_resource_roundtrip():
    """A wind resource dimensioned per turbine (one hub height each, no shared
    vertical profile) round-trips through dict_to_netcdf with the
    ``wind_turbine`` dimension and per-turbine height preserved."""
    res_yaml = _plant_examples_dir() / "plant_energy_resource" / "WTResource.yaml"

    windIO.validate(input=res_yaml, schema_type="plant/energy_resource")

    resource = windIO.load_yaml(res_yaml)
    ds = windIO.dict_to_netcdf(resource["wind_resource"])

    # Per-turbine resource: wind_turbine is a real dimension.
    assert "wind_turbine" in ds.dims
    assert ds.sizes["wind_turbine"] >= 2

    # Height is given per turbine (not a shared scalar / vertical profile).
    assert "height" in ds.variables
    assert ds["height"].dims == ("wind_turbine",)
    assert ds["height"].sizes["wind_turbine"] == ds.sizes["wind_turbine"]

    # The resource's data variables carry the per-turbine dimension (here
    # wind_speed / wind_direction are binned coordinate axes, so check the
    # actual per-turbine fields).
    per_turbine_vars = [
        v for v in ds.data_vars if "wind_turbine" in ds[v].dims
    ]
    assert per_turbine_vars, "no data variable carries the wind_turbine dimension"
    assert "sector_probability" in per_turbine_vars
