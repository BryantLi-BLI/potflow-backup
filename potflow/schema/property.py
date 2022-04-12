"""
Define the property (label) schema of an atomic configuration.
"""
from typing import Union

import numpy as np
import torch
from pydantic import BaseModel, Field


class Property(BaseModel):

    energy: Union[float, np.ndarray, torch.Tensor] = Field(
        None,
        title="Energy",
        description="Total energy of a configuration. Shape: (,)",
    )

    forces: Union[np.ndaray, torch.Tensor] = Field(
        None,
        title="Forces",
        description="Forces on atoms. These should be the total forces, not the "
        "partial forces. Shape: (N, 3), where N is the number of atoms in the "
        "configuration.",
    )

    stress: Union[np.ndaray, torch.Tensor] = Field(
        None,
        title="Stress",
        description="Stress on the simulation box. The stress is in the Voigt "
        "notation, with 6 components (i.e. Shape: (6,)). The 6 components are in the "
        "order of xx, yy, zz, yz, xz, and xy.",
    )
