import numpy as np

from isegeval.click.strategies import GreatestErrorRegion


def test_greatest_error_region() -> None:
    ground_truth_mask = np.zeros((500, 500), dtype=bool)
    ground_truth_mask[200:300, 200:300] = True

    strategy = GreatestErrorRegion(ground_truth_mask)

    segmentation_mask = np.zeros_like(ground_truth_mask)

    click = strategy.click(segmentation_mask)

    # 0-indexed
    assert click.x == 249
    assert click.y == 249

    assert click.is_positive
