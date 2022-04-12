from jobflow import JobStore, run_locally
from maggma.stores import MemoryStore, MongoStore
from pymatgen.core.structure import Structure

from potflow.labeling.vasp.flows import RelaxMaker
from potflow.labeling.vasp.jobs import StrainedStructureMaker


def get_job_store():
    memory_store1 = MemoryStore("fotflow")
    memory_store2 = MemoryStore("fotflow_data")
    store = JobStore(memory_store1, additional_stores={"data": memory_store2})

    return store


def get_strained_structure_wf(structure):
    maker = RelaxMaker(structure_composer=StrainedStructureMaker)
    flow = maker.make(structure)

    return flow


if __name__ == "__main__":

    # construct a rock salt MgO structure
    mgo_structure = Structure(
        lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
        species=["Mg", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    # create flow and store
    flow = get_strained_structure_wf(mgo_structure)
    store = get_job_store()

    # run the flow
    run_locally(flow, store=store)
