"""Defines constants for general use."""

import pathlib

local_dir = pathlib.Path(__file__).parent
root_dir = local_dir.parent

data_dir = root_dir / "data"
