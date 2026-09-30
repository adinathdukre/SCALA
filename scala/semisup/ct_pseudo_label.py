import argparse
import glob
import os
import shutil
import tempfile

import numpy as np
import SimpleITK as sitk

from .. import config as C
from ..data import io as IO
from ..postprocess.cc import postprocess_cardiac


def _bbox_with_margin(fg, spacing, margin_mm, shape):
    idx = np.argwhere(fg > 0)
    lo = idx.min(0)
    hi = idx.max(0) + 1
    mar = np.array([int(round(margin_mm / spacing[a])) for a in range(3)])
    lo = np.maximum(lo - mar, 0)
    hi = np.minimum(hi + mar, np.array(shape))
    return [int(x) for x in lo], [int(x) for x in (hi - lo)]


def pseudo_label(margin_mm, min_la_vox, stage_dir):
    from ..inference.predictor import build_predictor, predict_folder
    cases = []
    for i in range(51, 151):
        d = os.path.join(C.TASK_DIRS["cardiac"], f"train_{i}")
        if os.path.isdir(d) and IO.find_label(d, "cardiac") is None:
            cases.append((C.cardiac_caseid(i), IO.find_image(d)))
    img_out = os.path.join(stage_dir, "imagesTr")
    lab_out = os.path.join(stage_dir, "labelsTr")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)
    pred = build_predictor(C.model_dir(C.NAME_CARDIAC), tta=False)
    tmp = tempfile.mkdtemp(prefix="pseudo_", dir=os.environ.get("TMPDIR", tempfile.gettempdir()))
    kept, dropped = 0, []
    try:
        res = predict_folder(pred, cases, tmp, n_proc=2)
        for cid, img_path in cases:
            seg_sitk = sitk.ReadImage(res[cid])
            clean = postprocess_cardiac(IO.to_np(seg_sitk))
            la_vox = int((clean == 1).sum())
            if la_vox < min_la_vox:
                dropped.append(cid)
                continue
            lo, sz = _bbox_with_margin(clean > 0, seg_sitk.GetSpacing(), margin_mm, clean.shape)
            img_c = sitk.RegionOfInterest(sitk.ReadImage(img_path), sz, lo)
            lab_c = sitk.RegionOfInterest(IO.from_np(clean, seg_sitk), sz, lo)
            sitk.WriteImage(img_c, os.path.join(img_out, f"{cid}_0000.nii.gz"), useCompression=True)
            sitk.WriteImage(sitk.Cast(lab_c, sitk.sitkUInt8), os.path.join(lab_out, f"{cid}.nii.gz"),
                            useCompression=True)
            kept += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"kept {kept} pseudo-labeled CTs, dropped {len(dropped)}: {dropped}")


def build_dataset(stage_dir):
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    src = os.path.join(C.NNUNET_RAW, C.NAME_CARDIAC)
    dst = os.path.join(C.NNUNET_RAW, C.NAME_CARDIAC_SSL)
    n = 0
    for root in (src, stage_dir):
        for f in sorted(glob.glob(os.path.join(root, "imagesTr", "*_0000.nii.gz"))):
            cid = os.path.basename(f)[:-len("_0000.nii.gz")]
            lab = os.path.join(root, "labelsTr", f"{cid}.nii.gz")
            if not os.path.exists(lab):
                continue
            os.makedirs(os.path.join(dst, "imagesTr"), exist_ok=True)
            os.makedirs(os.path.join(dst, "labelsTr"), exist_ok=True)
            shutil.copy(f, os.path.join(dst, "imagesTr", os.path.basename(f)))
            shutil.copy(lab, os.path.join(dst, "labelsTr", f"{cid}.nii.gz"))
            n += 1
    generate_dataset_json(dst, {0: "CT"}, {"background": 0, "LA": 1, "LAA": 2, "PV": 3}, n, ".nii.gz",
                          dataset_name=C.NAME_CARDIAC_SSL)
    print(f"{C.NAME_CARDIAC_SSL}: {n} cases -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-mm", type=float, default=20.0)
    ap.add_argument("--min-la-vox", type=int, default=3000)
    ap.add_argument("--skip-predict", action="store_true")
    a = ap.parse_args()
    C.export_nnunet_env()
    stage = os.path.join(C.WORK_ROOT, "ct_pseudo")
    if not a.skip_predict:
        pseudo_label(a.margin_mm, a.min_la_vox, stage)
    build_dataset(stage)


if __name__ == "__main__":
    main()
