"""Tests for opt-in array-backed loading of ``!include`` netCDF resources.

``load_yaml(..., nc_data="array")`` keeps included netCDF data as numpy arrays
instead of nested Python lists (much cheaper for large resources), and
``validate(..., array_data=True)`` validates such inputs structure-only.
"""

from pathlib import Path

import numpy as np

import windIO

_RESOURCE = (
    Path(windIO.plant_ex.__file__).parent
    / "plant_energy_resource"
    / "WTResource_nc.yaml"
)
_SCHEMA = "plant/energy_resource"


def _first_var(wind_resource):
    return next(
        k
        for k, v in wind_resource.items()
        if isinstance(v, dict) and "data" in v and "dims" in v
    )


def test_default_loads_lists():
    """Default behaviour is unchanged: included netCDF data are Python lists."""
    wr = windIO.load_yaml(_RESOURCE)["wind_resource"]
    assert isinstance(wr[_first_var(wr)]["data"], list)


def test_array_mode_keeps_ndarrays_with_same_values():
    wr_arr = windIO.load_yaml(_RESOURCE, nc_data="array")["wind_resource"]
    wr_list = windIO.load_yaml(_RESOURCE)["wind_resource"]
    var = _first_var(wr_arr)

    assert isinstance(wr_arr[var]["data"], np.ndarray)
    # dims are preserved and values are identical to the list-backed load.
    assert list(wr_arr[var]["dims"]) == list(wr_list[var]["dims"])
    np.testing.assert_allclose(
        np.asarray(wr_list[var]["data"]), wr_arr[var]["data"]
    )


def test_structure_only_validation_file_and_dict():
    # Full validation still works.
    windIO.validate(input=_RESOURCE, schema_type=_SCHEMA)
    # Structure-only validation of the file (loads arrays, skips bulk data).
    windIO.validate(input=_RESOURCE, schema_type=_SCHEMA, array_data=True)
    # Structure-only validation of an already array-backed dict.
    data = windIO.load_yaml(_RESOURCE, nc_data="array")
    windIO.validate(input=data, schema_type=_SCHEMA, array_data=True)
