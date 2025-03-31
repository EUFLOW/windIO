
windIO: a community-focused data I/O format
===========================================

The repository defines two frameworks defining the inputs and outputs for systems engineering
MDAO of wind turbine and plants.
The github repository is at https://github.com/IEAWindSystems/windIO and the online documentation is at https://ieawindsystems.github.io/windIO.
The frameworks are implemented in two yaml-schemas, one for the turbine and one for the plant,
to enforce an ontology of how data should be organized.
The reference wind turbines designed within the IEA Wind Systems Engineering Task
are checked against the schema through unit testing.
You can access the yaml files at the following links:

- `IEA onshore 3.4 MW  <https://github.com/IEAWindTask37/IEA-3.4-130-RWT/blob/master/yaml/IEA-3.4-130-RWT.yaml>`_
- `IEA offshore 10.0 MW  <https://github.com/IEAWindTask37/IEA-10.0-198-RWT/blob/master/yaml/IEA-10-198-RWT.yaml>`_
- `IEA floating 15.0 MW  <https://github.com/IEAWindTask37/IEA-15-240-RWT/blob/master/WT_Ontology/IEA-15-240-RWT.yaml>`_
- `IEA 22-MW offshore <https://github.com/IEAWindSystems/IEA-22-280-RWT>`_
- `IEA Wind 740-10-MW Reference Offshore Wind Plants <https://github.com/IEAWindSystems/IEA-Wind-740-10-ROWP/blob/main/README.md>`_

   Author: `IEA Wind Task 37 Team <mailto:pietro.bortolotti@nrel.gov>`_

If you use this model in your research or publications, please cite this [IEA technical report](https://doi.org/10.2172/1868328):

    @article{osti_1868328,
    title = {System Modeling Frameworks for Wind Turbines and Plants: Review and Requirements Specifications},
    author = {Bortolotti, Pietro and Bay, Christopher and Barter, Garrett and Gaertner, Evan and Dykes, Katherine and McWilliam, Michael and Friis-Moller, Mikkel and Molgaard Pedersen, Mads and Zahle, Frederik},
    doi = {10.2172/1868328},
    place = {United States},
    year = {2022},
    month = {5}
    }

Installation
------------

windIO is typically included as a dependency in software that uses the windIO data format, so
users will normally not need to install it directly.
However, it can be useful to install the windIO package to access version converters or during
integration into a software package.
In that case, windIO can be installed from PyPI with the following command:

.. code-block:: bash

   pip install windIO

Supporting windIO in your software
----------------------------------

The windIO data format is defined by the schemas included in this repository.
In order for a software to support windIO, it must support the data as described in the schemas
and use the included functions to validate the data.
windIO should be included as a dependency.
It is distributed through PyPI and can be installed as a package with pip.
The suggested method of incorporating windIO into your code is:

.. code-block:: python

   import windIO

   # Other code here

   windIO.validate(input="path/to/input.yaml", schema_type="plant/wind_energy_system <for example>")
   windIO.load_yaml("path/to/input.yaml")

   # Conversion to your software's data structures here
