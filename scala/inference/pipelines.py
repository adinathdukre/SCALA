import os
import shutil
import tempfile
import traceback

import numpy as np
import SimpleITK as sitk

from ..data import io as IO
from ..postprocess.cc import (postprocess_cardiac_pvhyst, postprocess_cavity, remove_scar_off_blob,
                              remove_small)
from .predictor import build_predictor, predict_files, predict_one
from .probability import align_probabilities
from .scar import n_phases, predict_scar_prob_phases

SCAR_THRESHOLD = 0.20
SCAR_KEEP_MM = 6.0
PV_T_HI = 0.5
PV_T_LO = 0.25


def _scratch():
    return os.environ.get("TMPDIR", tempfile.gettempdir())


def fuse_softmax(preds, img, name):
    tmp = tempfile.mkdtemp(prefix="ens_", dir=_scratch())
    try:
        fused, ref = None, None
        for k, predictor in enumerate(preds):
            dst = os.path.join(tmp, f"{name}_0000.nii.gz")
            shutil.copy(img, dst)
            trunc = os.path.join(tmp, f"m{k}")
            predict_files(predictor, [[dst]], [trunc], save_probabilities=True)
            seg = IO.to_np(IO.read(trunc + ".nii.gz"))
            prob = align_probabilities(np.load(trunc + ".npz")["probabilities"], seg, source=trunc)
            ref = sitk.ReadImage(trunc + ".nii.gz")
            fused = prob if fused is None else fused + prob
        return fused / len(preds), ref
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_empty(img_path, out_path):
    image = sitk.ReadImage(img_path)
    IO.save_like(np.zeros(image.GetSize(), dtype=np.uint8), image, out_path, dtype=np.uint8)


def run_cases(cases, out_path_fn, label, body):
    print(f"[{label}] {len(cases)} cases", flush=True)
    failed = []
    for name, img in cases:
        op = out_path_fn(name)
        try:
            body(name, img, op)
            print(f"  ok {name} -> {op}", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            try:
                _write_empty(img, op)
            except Exception:
                traceback.print_exc()
    print(f"[{label}] {len(cases) - len(failed)}/{len(cases)} succeeded", flush=True)
    max_frac = float(os.environ.get("SCALA_MAX_FAIL_FRAC", "0.1"))
    if cases and len(failed) > max(0, int(max_frac * len(cases))):
        raise SystemExit(f"{len(failed)}/{len(cases)} cases failed and were written empty: {failed[:8]}")


def cavity_ensemble(cases, out_path_fn, model_dirs, folds=(0, 1, 2, 3, 4), tta=False, device="cuda"):
    preds = [build_predictor(m, folds=folds, tta=tta, device=device, on_device=False) for m in model_dirs]

    def body(name, img, op):
        fused, ref = fuse_softmax(preds, img, name)
        IO.save_like(postprocess_cavity(np.argmax(fused, 0).astype(np.uint8)), ref, op, dtype=np.uint8)

    run_cases(cases, out_path_fn, "task2 cavity", body)


def ct_ensemble(cases, out_path_fn, model_dirs, folds=(0, 1, 2, 3, 4), tta=False, device="cuda",
                t_hi=PV_T_HI, t_lo=PV_T_LO):
    preds = [build_predictor(m, folds=folds, tta=tta, device=device) for m in model_dirs]

    def body(name, img, op):
        fused, ref = fuse_softmax(preds, img, name)
        lab = np.argmax(fused, 0).astype(np.uint8)
        out = postprocess_cardiac_pvhyst(lab, fused[3].astype(np.float32), t_hi, t_lo)
        IO.save_like(out, ref, op, dtype=np.uint8)

    run_cases(cases, out_path_fn, "task3 CT", body)


def scar_ensemble(cases, out_path_fn, cavity_dir, scar_dirs, folds=(0, 1, 2, 3, 4), tta=False,
                  device="cuda", thr=SCAR_THRESHOLD, keep_mm=SCAR_KEEP_MM, min_cc=0):
    cav_pred = build_predictor(cavity_dir, folds=folds, tta=tta, device=device)
    scar_preds = [build_predictor(m, folds=folds, tta=tta, device=device) for m in scar_dirs]
    weight = np.float32(1.0 / len(scar_preds))

    def body(name, img, op):
        ref = IO.read(img)
        cav = postprocess_cavity(predict_one(cav_pred, img))
        fused = np.zeros(cav.shape, dtype=np.float32)
        for predictor, model_dir in zip(scar_preds, scar_dirs):
            fused += weight * predict_scar_prob_phases(ref, cav, predictor, n_phases(ref, model_dir))
        scar = (fused >= thr).astype(np.uint8)
        scar = remove_scar_off_blob(scar, (cav > 0).astype(np.uint8), ref.GetSpacing(), max_mm=keep_mm)
        if min_cc > 0:
            scar = remove_small(scar, min_cc)
        IO.save_like(scar, ref, op, dtype=np.uint8)

    run_cases(cases, out_path_fn, "task1 scar", body)
