from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker as Atomate2RelaxMaker
from jobflow import Flow, Maker, Response, job
from loguru import logger
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.analysis.elasticity import Strain
from pymatgen.core.structure import Structure
from pymatgen.core.tensors import symmetry_reduce
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder


from potflow import SETTINGS
from potflow._typing import Matrix6D
from potflow.labeling.vasp.jobs import StructureComposeMaker


@dataclass
class StructureComposeMaker(Maker):
    """
    A base Maker to compose (generate) structures.

    Args:
        relax: Whether to relax the generated structure.
        realx_maker: Atomate2 maker to relax the structure if `relax=True`.
        name: name of the job.
    """

    relax: bool = True
    relax_maker: BaseVaspMaker = field(default_factory=Atomate2RelaxMaker)
    name: str = "structure compose"

    @job
    def make(self, structure: Structure | tuple[Structure, ...]) -> list[Structure]:
        """
        Perform transformations on input structure to generate new structures.

        The actual transformation should be done in the `compose_structure()` method.
        This make function is a wrapper to make turn it into a job.

        Args:
            structure: Input pymatgen structure.

        Returns:
            Composed new structures.
        """
        generated_structure = self.compose_structure(structure)

        if isinstance(generated_structure, Structure):
            generated_structure = [generated_structure]

        if self.relax:
            static_jobs = [self.relax_maker.make(s) for s in generated_structure]
            outputs = [j.output.structure for j in static_jobs]
            flow = Flow(jobs=static_jobs, output=outputs)
            return Response(replace=flow)
        else:
            return generated_structure

    @abc.abstractmethod
    def compose_structure(
        self, structure: Structure | tuple[Structure, ...]
    ) -> Structure | list[Structure]:
        """
        Generate children structures from the input given parent structures.

        Args:
            structure: Input pymatgen structure.

        Returns:
            Generated new structure(s).
        """


@dataclass
class CoherentStructureMaker(StructureComposeMaker):
    """
    Maker to generate multiple strained structures.

    Args:
        interface_gap: Distance between two slabs when interfaced (default: 2 angstroms)
        slab1_thickness: Thickness of structure 1 slab; scaled by how many unit cells. Note this value corresponds to a multiplier.
            e.g slab1_thickness: 2 implies two unit cell of slab1 is used in the interfaced structure. (default: 1)
        slab2_thickness: Same properties as slab1, except for second slab structure
        slab1_miller_indices_upper_limit: highest miller indices value
            e.g slab1_miller_indices_upper_limit:3 yields only miller indices matches below (333) (default: 2)
        slab2_miller_indices_upper_limit: Same properties as slab1, except for second slab structure
    """

    name: str = "coherent interface structure job"
    interface_gap: float = 2
    slab1_thickness: float = 1
    slab2_thickness: float = 1
    slab1_miller_indices_upper_limit: int = 2
    slab2_miller_indices_upper_limit: int = 2
    miller_indices_matches: list[tuple] = None

    def miller_matches(self, struc1, struc2) -> list[miller_indices_matches]:
        sa = SubstrateAnalyzer(
            self.slab1_miller_indices_upper_limit, self.slab2_miller_indices_upper_limit
        )
        matches = list(sa.calculate(substrate=struc1, film=struc2))
        new_match = []
        for match in matches:
            new_match.append((match.film_miller, match.substrate_miller))
        return set(new_match)

    def compose_structure(
        self, structure: tuple[Structure, Structure]
    ) -> list[Structure]:
        """
        Generate the newy structures.

        Args:
            structure: parent structure.

        Returns:
            Strained structures generated from the parent structure.
        """

        structure1 = structure[0]
        structure2 = structure[1]

        struc1, struc2 = get_conventional_strucs(structure1, structure2)

        miller_matches = self.miller_matches(struc1, struc2)
        all_interfaces = []
        for millerindex in miller_matches:
            cib = CoherentInterfaceBuilder(
                substrate_structure=struc1,
                film_structure=struc2,
                film_miller=millerindex[0],
                substrate_miller=millerindex[1],
            )
            for termination in cib.terminations:
                interfaces = list(
                    cib.get_interfaces(
                        termination,
                        gap=self.interface_gap,
                        vacuum_over_film=self.interface_gap,
                    )
                )  # TODO sometimes creates duplicates
                all_interfaces.extend(interfaces)

        return all_interfaces


def get_conventional_strucs(struc1, struc2):
    # struc1, struc2 are structures directly obtained from MP via MPRester (eg. li_struct = mpr.get_structure_by_material_id('mp-135'))

    struc1_conventional = SpacegroupAnalyzer(
        struc1
    ).get_conventional_standard_structure()
    struc2_conventional = SpacegroupAnalyzer(
        struc2
    ).get_conventional_standard_structure()

    return struc1_conventional, struc2_conventional


# TODO, can it be generalized to use pymatgen AbstractTransformation to abstract this?
#  Currently, the transformation only accepts a single structure. We need to extend
#  it to accept multiple structures?
#  - The benefits of doing this is that we can keep track of what transformations are
#  applied to structure(s) and store them in a DB.
@dataclass
class StrainedStructureMaker(StructureComposeMaker):
    """
    Maker to generate multiple strained structures.

    Args:
        strain_states: Tuple of Voigt-notation strains. By default, the strains are
            along the x, y, z, yz, xz, and xy directions, i.e. the strain_states are:
            ``((1, 0, 0, 0, 0, 0),
               (0, 1, 0, 0, 0, 0),
               (0, 0, 1, 0, 0, 0),
               (0, 0, 0, 2, 0, 0),
               (0, 0, 0, 0, 2, 0),
               (0, 0, 0, 0, 0, 2))``.
        strain_magnitudes: A list of strain magnitudes to multiply by for each strain
            state, e.g. ``[-0.01, -0.005, 0.005, 0.01]``. Alternatively, a list of
            lists can be provided, where each inner list specifies the magnitude for
            each strain state.
        conventional: Whether to transform the structure into the conventional cell.
        sym_reduce: Whether to reduce the number of deformations using symmetry.
        symprec: Symmetry precision.
    """

    name: str = "strained structure job"
    strain_states: Matrix6D = (
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 2, 0, 0),
        (0, 0, 0, 0, 2, 0),
        (0, 0, 0, 0, 0, 2),
    )
    strain_magnitudes: float | list[float] | list[list[float]] = field(
        default_factory=lambda: [-0.01, -0.005, 0.05, 0.01]
    )
    conventional: bool = False
    sym_reduce: bool = True
    symprec: float = SETTINGS.SYMPREC

    def compose_structure(self, structure: Structure) -> list[Structure]:
        """
        Generate the new structures.

        Args:
            structure: parent structure.

        Returns:
            Strained structures generated from the parent structure.
        """
        if self.conventional:
            sga = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = sga.get_conventional_standard_structure()

        strain_magnitudes = self.strain_magnitudes
        if np.asarray(strain_magnitudes).ndim == 1:
            strain_magnitudes = [strain_magnitudes] * len(self.strain_states)

        strains = []
        for state, magnitudes in zip(self.strain_states, strain_magnitudes):
            strains.extend(
                [Strain.from_voigt(m * np.asarray(state)) for m in magnitudes]
            )

        # remove zero strains
        strains = [s for s in strains if (abs(s) > 1e-10).any()]
        deformations = [s.get_deformation_matrix() for s in strains]

        if self.sym_reduce:
            deformation_mapping = symmetry_reduce(
                deformations, structure, symprec=self.symprec
            )
            logger.info(
                "Using symmetry to reduce number of deformations from "
                f"{len(deformations)} to {len(list(deformation_mapping.keys()))}"
            )
            deformations = list(deformation_mapping.keys())

        # strain the structure
        deformed_structures: list[Structure] = []
        for deform in deformations:
            dst = DeformStructureTransformation(deformation=deform)
            ts = TransformedStructure(structure, transformations=[dst])
            deformed_structures.append(ts.final_structure)

        return deformed_structures
