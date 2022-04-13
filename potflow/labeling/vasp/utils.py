from atomate2.vasp.schemas.task import TaskDocument
from jobflow import job
from pymatgen.core.structure import Structure

from potflow.schema.configuration import Configuration, Property


@job(output_schmea=Configuration)
def task_doc_to_configuration(task_doc: TaskDocument):
    structure = Structure.from_dict(task_doc["structure"])
    config = Configuration.from_pymatgen_structure(structure)

    return config


@job(output_schmea=Configuration)
def task_doc_to_property(task_doc: TaskDocument):
    data = {
        "energy": task_doc["energy"],
        "forces": task_doc["forces"],
        "stress": task_doc["stress"],
    }

    return data
