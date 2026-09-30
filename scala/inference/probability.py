from itertools import permutations

import numpy as np


def align_probabilities(probability: np.ndarray, segmentation: np.ndarray, source: str = "") -> np.ndarray:
    probability = np.asarray(probability)
    segmentation = np.asarray(segmentation)
    if probability.ndim != segmentation.ndim + 1:
        raise ValueError(f"probability rank {probability.ndim} incompatible with segmentation ({source})")
    canonical = (0, *range(probability.ndim - 1, 0, -1))
    candidates = []
    for spatial in permutations(range(1, probability.ndim)):
        axes = (0, *spatial)
        if tuple(probability.shape[i] for i in spatial) != segmentation.shape:
            continue
        aligned = np.ascontiguousarray(probability.transpose(axes))
        argmax = np.argmax(aligned, axis=0)
        overall = float(np.mean(argmax == segmentation))
        union = (argmax != 0) | (segmentation != 0)
        foreground = float(np.mean(argmax[union] == segmentation[union])) if union.any() else 1.0
        exact = bool(np.array_equal(argmax, segmentation))
        candidates.append(((int(exact), foreground, overall, int(axes == canonical)), axes, aligned))
    if not candidates:
        raise ValueError(f"no permutation maps {probability.shape} to {segmentation.shape} ({source})")
    score, axes, aligned = max(candidates, key=lambda item: item[0])
    if not score[0] and (score[1] < 0.98 or score[2] < 0.98):
        raise ValueError(f"cannot align probability {probability.shape} to {segmentation.shape} ({source})")
    return aligned


def to_xyz(prob: np.ndarray, ref_shape) -> np.ndarray:
    if prob.shape == tuple(ref_shape):
        return prob
    if prob.shape == tuple(ref_shape)[::-1]:
        return prob.transpose(2, 1, 0)
    raise ValueError(f"prob shape {prob.shape} incompatible with ref {ref_shape}")
