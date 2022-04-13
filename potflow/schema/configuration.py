"""
Define the atomic configuration.
"""

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure

from potflow._typing import Vector
from potflow.schema.property import Property
from potflow.utils.uuid import suuid


class Configuration(BaseModel):
    """
    An atomic configuration.
    """

    species: Sequence[str] = Field(
        title="Species",
        description="Species of the atoms, typically their atomic symbols",
    )

    coords: Union[np.ndaray, torch.Tensor, List[Vector]] = Field(
        title="Coords",
        description="Coordinates of the atoms. Shape (N, 3), where N is the number of "
        "atoms in the configuration.",
    )

    cell: Tuple[Vector, Vector, Vector] = Field(
        None,
        title="Cell",
        description="Cell vectors a_1, a_2, and a_3 of the simulation box. If `None`, "
        "this is a cluster without cell (typical for a molecule).",
    )

    PBC: Union[Tuple[bool, bool, bool], Tuple[int, int, int]] = Field(
        (True, True, True),
        title="PBC",
        description="Periodic boundary conditions along the three cell vectors a_1, "
        "a_2, and a_3.",
    )

    # TODO, maybe not do this
    property: Property = Field(
        None,
        title="Property",
        description="Properties associated with the configuration.",
    )

    uuid: str = Field(
        default_factory=suuid, title="UUID", description="A string " "UUID."
    )

    @classmethod
    def from_ase(cls):
        pass

    @classmethod
    def from_pymatgen_structure(cls, structure: Structure):
        pass

    @classmethod
    def from_colabfit(cls):
        pass
