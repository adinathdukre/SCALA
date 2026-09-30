import json
import math
import os
import shutil
import tempfile

import numpy as np
import SimpleITK as sitk

from .. import config as C
from ..data import io as IO
from ..data.wallband import scar_channels
from .predictor import predict_files
from .probability import to_xyz


def predict_scar_prob(ref_sitk, cavity_arr, predictor, roi_mm=C.ROI_MM_MRI):
    lge = IO.to_np(ref_sitk).astype(np.float32)
    cav = (cavity_arr > 0).astype(np.uint8)
    probfull = np.zeros(lge.shape, np.float32)
    if cav.sum() < 50:
        return probfull
    sl, chans = scar_channels(lge, cav, ref_sitk.GetSpacing(), roi_mm)
    roi_ref = sitk.RegionOfInterest(ref_sitk, [s.stop - s.start for s in sl], [s.start for s in sl])
    tmp = tempfile.mkdtemp(prefix="scarprob_")
    try:
        paths = []
        for ch, arr in enumerate(chans):
            p = os.path.join(tmp, f"case_{ch:04d}.nii.gz")
            IO.save_like(arr.astype(np.float32), roi_ref, p, dtype=np.float32)
            paths.append(p)
        predict_files(predictor, [paths], [os.path.join(tmp, "case")], save_probabilities=True)
        prob = np.load(os.path.join(tmp, "case.npz"))["probabilities"][1]
        probfull[sl] = to_xyz(prob, chans[0].shape).astype(np.float32)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return probfull


def _shift(arr, p, back=False):
    if p == 0:
        return arr
    out = np.empty_like(arr)
    if back:
        out[..., p:] = arr[..., :-p]
        out[..., :p] = arr[..., :1]
    else:
        out[..., :-p] = arr[..., p:]
        out[..., -p:] = arr[..., -1:]
    return out


def n_phases(ref_sitk, model_dir, cap=3):
    try:
        with open(os.path.join(model_dir, "plans.json")) as f:
            plan_z = float(json.load(f)["configurations"][C.CONFIGURATION]["spacing"][0])
        ratio = plan_z / float(ref_sitk.GetSpacing()[2])
    except Exception:
        return 1
    return int(max(1, min(cap, math.ceil(ratio - 1e-6))))


def predict_scar_prob_phases(ref_sitk, cavity_arr, predictor, phases=1):
    if phases <= 1:
        return predict_scar_prob(ref_sitk, cavity_arr, predictor)
    probs = []
    for p in range(phases):
        img_p = IO.from_np(_shift(IO.to_np(ref_sitk), p), ref_sitk)
        prob = predict_scar_prob(img_p, _shift(cavity_arr, p), predictor)
        probs.append(_shift(prob, p, back=True))
    return np.median(np.stack(probs, 0), axis=0)
