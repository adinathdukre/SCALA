import os
import shutil
import tempfile

import torch

_SHM_MIN_MB = 512
_sequential = None


def build_predictor(model_dir, folds=(0, 1, 2, 3, 4), checkpoint="checkpoint_best.pth",
                    tta=False, step=0.5, device="cuda", on_device=None):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    if on_device is None:
        on_device = device == "cuda" and os.environ.get("SCALA_LOW_VRAM") != "1"
    predictor = nnUNetPredictor(
        tile_step_size=step, use_gaussian=True, use_mirroring=tta,
        perform_everything_on_device=on_device and device == "cuda",
        device=torch.device(device), verbose=False, verbose_preprocessing=False,
        allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=folds,
                                                   checkpoint_name=checkpoint)
    return predictor


def use_sequential():
    global _sequential
    if _sequential is not None:
        return _sequential
    forced = os.environ.get("SCALA_SEQUENTIAL")
    if forced in ("0", "1"):
        _sequential = forced == "1"
    else:
        try:
            st = os.statvfs("/dev/shm")
            mb = st.f_bsize * st.f_bavail / 2 ** 20
        except OSError:
            mb = 0.0
        _sequential = mb < _SHM_MIN_MB
    return _sequential


def predict_files(predictor, in_lists, out_truncs, save_probabilities=False, n_proc=1):
    if use_sequential():
        return predictor.predict_from_files_sequential(
            in_lists, out_truncs, save_probabilities=save_probabilities, overwrite=True)
    return predictor.predict_from_files(
        in_lists, out_truncs, save_probabilities=save_probabilities, overwrite=True,
        num_processes_preprocessing=n_proc, num_processes_segmentation_export=n_proc)


def predict_folder(predictor, cases, out_dir, n_proc=2):
    os.makedirs(out_dir, exist_ok=True)
    tmp_in = tempfile.mkdtemp(prefix="nnin_")
    in_lists, out_trunc = [], []
    for name, img in cases:
        dst = os.path.join(tmp_in, f"{name}_0000.nii.gz")
        shutil.copy(img, dst)
        in_lists.append([dst])
        out_trunc.append(os.path.join(out_dir, name))
    try:
        predict_files(predictor, in_lists, out_trunc, save_probabilities=False, n_proc=n_proc)
    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)
    return {name: os.path.join(out_dir, f"{name}.nii.gz") for name, _ in cases}


def predict_one(predictor, image_path):
    import SimpleITK as sitk
    from ..data import io as IO
    tmp = tempfile.mkdtemp(prefix="nnone_")
    try:
        res = predict_folder(predictor, [("case", image_path)], tmp, n_proc=1)
        return IO.to_np(sitk.ReadImage(res["case"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
