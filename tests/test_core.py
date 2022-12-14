import numpy as np

from isegeval.click import Click
from isegeval.core import evaluate


class Model:
    def __init__(self, ground_truth_mask: np.ndarray) -> None:
        self.ground_truth_mask = ground_truth_mask

    def predict(self, click: Click) -> np.ndarray:
        return self.ground_truth_mask


class ModelFactory:
    def __init__(self, ground_truth_mask: np.ndarray) -> None:
        self.ground_truth_mask = ground_truth_mask

    def get_model(self, image: np.ndarray) -> Model:
        return Model(self.ground_truth_mask)


def test_evaluate() -> None:
    image = np.random.rand(500, 500, 3)
    ground_truth_mask = np.zeros((500, 500), dtype=bool)
    ground_truth_mask[200:300, 200:300] = True

    noc = evaluate([(image, ground_truth_mask)], ModelFactory(ground_truth_mask))

    assert noc == 1.0
