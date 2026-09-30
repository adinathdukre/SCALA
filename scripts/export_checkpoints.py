import argparse
import os
import shutil

import torch

MEMBERS = {
    "cavity_resenc": ("Dataset501_LAcavity", "nnUNetTrainer"),
    "cavity_mednext": ("Dataset501_LAcavity", "nnUNetTrainerMedNeXt"),
    "scar_resenc": ("Dataset512_LAscarSSL", "nnUNetTrainer"),
    "scar_surface": ("Dataset512_LAscarSSL", "nnUNetTrainerScarSurface"),
    "ct_resenc": ("Dataset522_CardiacCTssl", "nnUNetTrainer"),
    "ct_stunet": ("Dataset522_CardiacCTssl", "STUNetTrainer_base_ft"),
}
DROP = ("optimizer_state", "grad_scaler_state", "logging")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("nnUNet_results"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    a = ap.parse_args()
    for name, (dataset, trainer) in MEMBERS.items():
        src = os.path.join(a.results, dataset, f"{trainer}__{a.plans}__3d_fullres")
        dst = os.path.join(a.out, name)
        os.makedirs(dst, exist_ok=True)
        for f in ("plans.json", "dataset.json"):
            shutil.copy(os.path.join(src, f), dst)
        for fold in range(5):
            ck = torch.load(os.path.join(src, f"fold_{fold}", a.checkpoint), map_location="cpu", weights_only=False)
            os.makedirs(os.path.join(dst, f"fold_{fold}"), exist_ok=True)
            torch.save({k: v for k, v in ck.items() if k not in DROP},
                       os.path.join(dst, f"fold_{fold}", "checkpoint_best.pth"))
        print(f"{name}: {src} -> {dst}")


if __name__ == "__main__":
    main()
