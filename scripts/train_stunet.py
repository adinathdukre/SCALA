from unittest.mock import patch

import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
import nnunetv2.run.run_training as rt
from nnunetv2.run.run_training import run_training_entry


def load_stunet_pretrained_weights(network, fname, verbose=False):
    saved = torch.load(fname, map_location="cpu", weights_only=False)
    pretrained = saved["network_weights"] if str(fname).endswith("pth") else saved["state_dict"]
    mod = network.module if isinstance(network, DDP) else network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod
    model_dict = mod.state_dict()
    n_in = model_dict["conv_blocks_context.0.0.conv1.weight"].shape[1]
    if n_in > 1:
        for k in ("conv_blocks_context.0.0.conv1.weight", "conv_blocks_context.0.0.conv3.weight"):
            pretrained[k] = pretrained[k].repeat(1, n_in, 1, 1, 1)
    for key in model_dict:
        if "seg_outputs" not in key:
            assert key in pretrained, f"{key} missing in pretrained weights"
            assert model_dict[key].shape == pretrained[key].shape, f"shape mismatch for {key}"
    pretrained = {k: v for k, v in pretrained.items() if k in model_dict and "seg_outputs" not in k}
    model_dict.update(pretrained)
    mod.load_state_dict(model_dict)
    print(f"loaded {len(pretrained)} pretrained tensors from {fname}")


if __name__ == "__main__":
    with patch.object(rt, "load_pretrained_weights", load_stunet_pretrained_weights):
        run_training_entry()
