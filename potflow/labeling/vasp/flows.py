from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import RelaxMaker as Atomate2RelaxMaker
from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from potflow.labeling.vasp.jobs import StructureComposeMaker
from potflow.labeling.vasp.utils import task_doc_to_property, task_doc_to_configuration


@dataclass
class RelaxMaker(Maker):
    """
    Abstract class for data labeling.

    Args:
        structure_composer: A maker to generate structure(s) from the input parent
            structure, e.g. `StrainedStructureMaker`.
        structure_composer_kwargs: Keyword arguments passed to the structure_composer.
        relax_parent_structure: Whether to relax the input parent structure.
        relax_composed_structure: Whether to relax the generated structure(s).
        relax_parent_maker: Atomate2 maker to relax the parent structure, if
            `realx_parent_structure=True`. Default to
            `atomate2.vasp.jobs.core.RelaxMaker`. Other options include `StaticMaker`
            and `TightRelaxMaker` in `atomate2.vasp.jobs.core`,
            as well as `DoubleRelaxMaker` in `atomate2.vasp.flows.core`.
        relax_composed_maker: Atomate2 maker to relax the generated structure,
        if `relax_composed_structure=True`. Same default and other options as for
        `realx_parent_maker`.
        name: Name of the generated flow.
    """

    structure_composer: StructureComposeMaker = None
    structure_composer_kwargs: dict = field(default_factory=dict)
    relax_parent_structure: bool = True
    relax_composed_structure: bool = True
    relax_parent_maker: BaseVaspMaker = field(default_factory=Atomate2RelaxMaker)
    relax_composed_maker: BaseVaspMaker = field(default_factory=Atomate2RelaxMaker)
    name: str = "potflow relax"

    def make(self, structure: Structure, prev_vasp_dir: str | Path | None = None):
        jobs = []

        # relax parent structure
        if self.relax_parent_structure:
            j = self.relax_parent_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            jobs.append(j)
            j.name += "_parent"

            structure = j.output.structure
            prev_vasp_dir = j.output.dir_name

        # relax generated structure
        composer = self.structure_composer(**self.structure_composer_kwargs)
        generated_structure = composer.make(structure)
        jobs.append(generated_structure)

        flow = Flow(jobs=jobs, output=, name=self.name)

        return flow
