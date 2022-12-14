import numpy as np

from isegeval import utils


def test_get_iou() -> None:

    ground_truth_mask = np.zeros((500, 500), dtype=bool)
    ground_truth_mask[200:300, 200:300] = True

    segmentation_mask = np.zeros_like(ground_truth_mask)
    segmentation_mask[210:310, 210:310] = True

    assert utils.get_iou(ground_truth_mask, segmentation_mask) == 90**2 / (
        90**2 + 3800
    )


def test_compute_noc() -> None:

    ious = [
        np.array([0.7, 0.8, 0.82, 0.85, 0.89, 0.95]),
        np.array([0.7, 0.75, 0.82, 0.85, 0.89, 0.89, 0.89, 0.9]),
    ]

    assert utils.compute_noc(ious, 0.9) == 7.0
