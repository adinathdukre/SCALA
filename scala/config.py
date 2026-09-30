import glob
import os

RAW_ROOT = os.environ.get("SCALA_DATA", "/path/to/CARE-LeftAtrium2026/dataset")


def _task_dir(prefix):
    hits = sorted(glob.glob(os.path.join(RAW_ROOT, prefix + "*")))
    return os.path.join(hits[0] if hits else os.path.join(RAW_ROOT, prefix), "train_data")


TASK_DIRS = {
    "scar": _task_dir("LA scar quantification"),
    "cavity": _task_dir("LA cavity segmentation"),
    "cardiac": _task_dir("cardiac anatomy segmentation"),
}

NNUNET_RAW = os.environ.get("nnUNet_raw", "/path/to/nnUNet_raw")
NNUNET_PREPROCESSED = os.environ.get("nnUNet_preprocessed", "/path/to/nnUNet_preprocessed")
NNUNET_RESULTS = os.environ.get("nnUNet_results", "/path/to/nnUNet_results")
WORK_ROOT = os.environ.get("SCALA_WORK", "/path/to/work")
CAVITY_OOF = os.path.join(WORK_ROOT, "cavity_oof")

PLANS = "nnUNetResEncUNetLPlans"
CONFIGURATION = "3d_fullres"

NAME_CAVITY = "Dataset501_LAcavity"
NAME_SCAR = "Dataset511_LAscarROI"
NAME_SCAR_SSL = "Dataset512_LAscarSSL"
NAME_CARDIAC = "Dataset521_CardiacCT"
NAME_CARDIAC_SSL = "Dataset522_CardiacCTssl"

ROI_MM_MRI = (200.0, 140.0, 110.0)
WALL_DILATE_MM = 2.5

N_FOLDS = 5
SEED = 1234


def cavity_caseid(i: int) -> str:
    return f"LA_{i:03d}"


def cardiac_caseid(i: int) -> str:
    return f"CT_{i:03d}"


def model_dir(dataset: str, trainer: str = "nnUNetTrainer") -> str:
    return os.path.join(NNUNET_RESULTS, dataset, f"{trainer}__{PLANS}__{CONFIGURATION}")


def export_nnunet_env():
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"] = NNUNET_RESULTS
    os.makedirs(WORK_ROOT, exist_ok=True)
