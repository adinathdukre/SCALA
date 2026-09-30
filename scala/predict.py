import argparse
import glob
import os
import sys
import traceback

from .data import io as IO

TASKS = {
    "task1": {"cavity": "cavity_resenc", "scar": ["scar_resenc", "scar_surface"]},
    "task2": {"models": ["cavity_resenc", "cavity_mednext"]},
    "task3": {"models": ["ct_resenc", "ct_stunet"]},
}


def discover(input_dir, task):
    root = os.path.join(input_dir, task) if os.path.isdir(os.path.join(input_dir, task)) else input_dir
    nested = root != input_dir
    cases = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d):
            try:
                cases.append((os.path.basename(d), IO.find_image(d), True))
            except FileNotFoundError:
                print(f"no image in {d}, skipped")
        elif d.endswith(".nii.gz"):
            cases.append((os.path.basename(d)[:-len(".nii.gz")], d, False))
    return cases, nested


def output_fn(output_dir, task, nested, case_dirs):
    base = os.path.join(output_dir, task) if nested else output_dir

    def fn(name):
        if case_dirs.get(name):
            xx = name.split("_", 1)[1] if "_" in name else name
            return os.path.join(base, name, f"{xx}_pred.nii.gz")
        return os.path.join(base, f"{name}_pred.nii.gz")
    return fn


def selftest(model_dirs):
    from .inference.predictor import build_predictor
    for m in model_dirs:
        if not os.path.isdir(m):
            sys.exit(f"missing model directory {m}")
        build_predictor(m, device="cpu")
        print(f"loaded {m}", flush=True)
    print(f"selftest ok ({len(model_dirs)} models)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default=os.environ.get("SCALA_TASK"))
    ap.add_argument("-i", "--input", default=os.environ.get("INPUT_DIR", "/input"))
    ap.add_argument("-o", "--output", default=os.environ.get("OUTPUT_DIR", "/output"))
    ap.add_argument("-m", "--models", default=os.environ.get("SCALA_MODELS", "/models"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.task is None:
        ap.error("--task is required")
    spec = TASKS[a.task]
    folds = tuple(int(f) for f in a.folds.split(","))
    join = lambda names: [os.path.join(a.models, n) for n in names]

    if a.selftest:
        selftest(join([spec["cavity"]] + spec["scar"]) if a.task == "task1" else join(spec["models"]))
        return 0

    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("no CUDA device visible; run the container with --gpus all or pass --device cpu")

    cases, nested = discover(a.input, a.task)
    if not cases:
        sys.exit(f"no cases found under {a.input}")
    case_dirs = {name: is_dir for name, _, is_dir in cases}
    cases = [(name, path) for name, path, _ in cases]
    out_fn = output_fn(a.output, a.task, nested, case_dirs)

    from .inference import pipelines as P
    kw = dict(folds=folds, tta=a.tta, device=a.device)
    if a.task == "task1":
        P.scar_ensemble(cases, out_fn, join([spec["cavity"]])[0], join(spec["scar"]), **kw)
    elif a.task == "task2":
        P.cavity_ensemble(cases, out_fn, join(spec["models"]), **kw)
    else:
        P.ct_ensemble(cases, out_fn, join(spec["models"]), **kw)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main() or 0
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, flush=True)
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
