import argparse
import glob
import os
import shutil
import tempfile

import numpy as np
import SimpleITK as sitk

from .. import config as C
from ..data import io as IO
from ..data.scar_cavity_map import scar_to_cavity
from ..data.wallband import scar_channels
from ..postprocess.cc import postprocess_cavity, remove_scar_off_blob


def _cavity_only_ids():
    scar_ids = set(scar_to_cavity().values())
    img_dir = os.path.join(C.NNUNET_RAW, C.NAME_CAVITY, "imagesTr")
    out = []
    for lp in sorted(glob.glob(os.path.join(C.NNUNET_RAW, C.NAME_CAVITY, "labelsTr", "*.nii.gz"))):
        cid = os.path.basename(lp)[:-len(".nii.gz")]
        if cid in scar_ids:
            continue
        if round(sitk.ReadImage(os.path.join(img_dir, f"{cid}_0000.nii.gz")).GetSpacing()[2], 2) > 1.5:
            out.append(cid)
    return out


def _copy_real(img_d, lab_d):
    src = os.path.join(C.NNUNET_RAW, C.NAME_SCAR)
    n = 0
    for f in sorted(glob.glob(os.path.join(src, "labelsTr", "*.nii.gz"))):
        cid = os.path.basename(f)[:-len(".nii.gz")]
        for ch in range(3):
            shutil.copy(os.path.join(src, "imagesTr", f"{cid}_{ch:04d}.nii.gz"), img_d)
        shutil.copy(f, os.path.join(lab_d, f"{cid}.nii.gz"))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-vox", type=int, default=40)
    ap.add_argument("--max-vox", type=int, default=40000)
    a = ap.parse_args()
    C.export_nnunet_env()
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    from ..inference.predictor import build_predictor
    from ..inference.probability import to_xyz

    dst = os.path.join(C.NNUNET_RAW, C.NAME_SCAR_SSL)
    img_d = os.path.join(dst, "imagesTr")
    lab_d = os.path.join(dst, "labelsTr")
    os.makedirs(img_d, exist_ok=True)
    os.makedirs(lab_d, exist_ok=True)
    n_real = _copy_real(img_d, lab_d)

    pred = build_predictor(C.model_dir(C.NAME_SCAR), tta=False)
    cav_img = os.path.join(C.NNUNET_RAW, C.NAME_CAVITY, "imagesTr")
    stage = tempfile.mkdtemp(prefix="scarssl_", dir=os.environ.get("TMPDIR", tempfile.gettempdir()))
    meta, chan_lists, out_stems = {}, [], []
    for cid in _cavity_only_ids():
        oof = os.path.join(C.CAVITY_OOF, f"{cid}.nii.gz")
        if not os.path.exists(oof):
            continue
        ref = IO.read(os.path.join(cav_img, f"{cid}_0000.nii.gz"))
        cav = postprocess_cavity(IO.to_np(IO.match_geometry(IO.read(oof), ref)))
        if cav.sum() < 50:
            continue
        sl, chans = scar_channels(IO.to_np(ref).astype(np.float32), cav, ref.GetSpacing(), C.ROI_MM_MRI)
        roi_ref = sitk.RegionOfInterest(ref, [s.stop - s.start for s in sl], [s.start for s in sl])
        paths = []
        for ch, arr in enumerate(chans):
            p = os.path.join(stage, f"{cid}_{ch:04d}.nii.gz")
            IO.save_like(arr.astype(np.float32), roi_ref, p, dtype=np.float32)
            paths.append(p)
        chan_lists.append(paths)
        out_stems.append(os.path.join(stage, f"{cid}_pred"))
        meta[cid] = (roi_ref, (cav[sl] > 0).astype(np.uint8), ref.GetSpacing())

    pred.predict_from_files(chan_lists, out_stems, save_probabilities=True, overwrite=True,
                            num_processes_preprocessing=3, num_processes_segmentation_export=3)

    kept = []
    for cid, (roi_ref, cav_roi, spacing) in meta.items():
        seg = sitk.ReadImage(os.path.join(stage, f"{cid}_pred.nii.gz"))
        prob = to_xyz(np.load(os.path.join(stage, f"{cid}_pred.npz"))["probabilities"][1], IO.to_np(seg).shape)
        scar = remove_scar_off_blob((prob >= a.thr).astype(np.uint8), cav_roi, spacing,
                                    max_mm=C.WALL_DILATE_MM + 2)
        if not a.min_vox <= int(scar.sum()) <= a.max_vox:
            continue
        for ch in range(3):
            shutil.copy(os.path.join(stage, f"{cid}_{ch:04d}.nii.gz"), img_d)
        IO.save_like(scar, roi_ref, os.path.join(lab_d, f"{cid}.nii.gz"), dtype=np.uint8)
        kept.append(cid)
    shutil.rmtree(stage, ignore_errors=True)
    generate_dataset_json(dst, {0: "LGE", 1: "noNorm", 2: "noNorm"}, {"background": 0, "scar": 1},
                          n_real + len(kept), ".nii.gz", dataset_name=C.NAME_SCAR_SSL)
    print(f"{C.NAME_SCAR_SSL}: {n_real} labeled + {len(kept)} pseudo-labeled -> {dst}")


if __name__ == "__main__":
    main()
