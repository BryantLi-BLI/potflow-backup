import abc
from typing import Callable, Dict, Optional, Sequence, Tuple

from monty.json import MSONable
from torch import Tensor
from torchmetrics import MetricCollection


class Task(abc.ABC, MSONable):
    """
    Abstract class for a task that defines how to deal with loss and metrics.
    """

    def __init__(self, name: str):

        self._name = name
        self.loss_fn = self.get_loss_fn()
        self.metric_fn = self._get_metric_fn_as_collection()

    def compute_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        pass

    def compute_metric(self, pred: Tensor, target: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def get_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Returns the loss function for the task.

        The loss function should take two tensors as input and returns a float tensor.
            def loss_fn(pred, target) -> float:
                loss = ...
                return  loss
        """

    # TODO, make it torchmetric independent. also, _get_metric_fn_as_collection()
    @abc.abstractmethod
    def get_metric_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Returns the metric function (or a metric collection) for the task.

        The metric function should take two tensors as input and returns a float tensor.
            def metric_fn(pred, target) -> float:
                metric = ...
                return  metric

        Example:
            metric = Accuracy(num_classes=10)
            return metric

        Instead of a metric function, it could also be a metric collection.

        Example:
            num_classes = 10
            metric = MetricCollection(
                [
                    Accuracy(num_classes=10),
                    F1(num_classes=10),
                ]
            )
            return metric
        """

    def _get_metric_fn_as_collection(self) -> Callable[[Tensor, Tensor], Tensor]:
        """
        This is a wrapper for `get_metric_fn()` to convert metric function to a
        metric collection.
        """
        metric_fn = self.get_metric_fn()
        if not isinstance(metric_fn, MetricCollection):
            metric_fn = MetricCollection([metric_fn])

        return metric_fn

    @property
    def name(self):
        return self._name

    def transform_for_loss(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Transform the task predictions and labels.

        For regression task, we may scale the target by subtracting the mean and
        then dividing the standard deviation. To recover the prediction in the
        original space, we should reverse the process. Inverse transformation like this
        can be implemented here.

        Args:
            pred: model prediction
            target: reference target for the prediction

        Returns:
            transformed_pred
            transformed_target
        """

        return pred, target

    def transform_for_metric(
        self, pred: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:

        return pred, target


class TaskCollection:
    """
    A collection of tasks.

    This is a wrapper to make working with multiple tasks easier.

    Args:
        tasks: tasks for the model
        loss_weight:
        metric_weight:
    """

    def __init__(self, tasks: Sequence[Task], loss_weight=None, metric_weight=None):
        self.tasks = tasks
        self.loss_weight = loss_weight
        self.metric_weight = metric_weight

    def compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:

        if weight is None:
            weight = {t.name for t in self.tasks}

        losses = {
            t.name: t.compute_loss(pred, target, weight[t.name]) for t in self.tasks
        }
