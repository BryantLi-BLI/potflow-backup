from abc import ABC, abstractmethod
from typing import Tuple

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_analyzer import


class Labeler(ABC):
    """
    Abstract class for data labeling.

    Args:
        structure: structure for the labeler to label
    """

    def __init__(self, structure: Structure):
        self.structure = structure

    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def job(self):
        """
        Jobflow job that uses the structure to generate the output.
        """
        pass


class StrainedLabeler(Labeler):
    def __init__(
        self,
        structure: Structure,
        a_range: Tuple[float, float],
        b_range: Tuple[float, float],
        c_range: Tuple[float, float],
        a_step: float = 0.01,
        b_step: float = 0.01,
        c_step: float = 0.01,
    ):
        """
        Generate a set of strained structures for labeling.

        Args:
            structure:
            a_range:
            b_range:
            c_range:
            a_step:
            b_step:
            c_step:
        """
        super().__init__(structure)

    def output(self):
        pass

    def job(self):
        pass

def strain_structures(structure:Structure,
                      a_range: Tuple[float, float],
                      b_range: Tuple[float, float],
                      c_range: Tuple[float, float],
                      a_step: float = 0.01,
                      b_step: float = 0.01,
                      c_step: float = 0.01 ):
    # symmetry analysis


#
# if __name__ == "__main__":
#     lb = StaticLabeler()
#     print(lb)
