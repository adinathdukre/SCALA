import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerScarSurface(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.surf_sigma_mm = 2.0
        self.surf_weight = 0.5

    def train_step(self, batch: dict) -> dict:
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _b
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([i.to(self.device, non_blocking=True) for i in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        self.optimizer.zero_grad(set_to_none=True)
        ctx = _b.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else _b.dummy_context()
        with ctx:
            output = self.network(data)
            loss = self.loss(output, target)
            o0 = output[0] if isinstance(output, (list, tuple)) else output
            t0 = target[0] if isinstance(target, (list, tuple)) else target
            w = torch.exp(-(data[:, 2:3] / self.surf_sigma_mm) ** 2 / 2.0)
            p = torch.softmax(o0.float(), 1)[:, 1:2]
            tgt = (t0 == 1).float()
            num = 2.0 * (w * p * tgt).sum() + 1.0
            den = (w * p).sum() + (w * tgt).sum() + 1.0
            loss = loss + self.surf_weight * (1.0 - num / den)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}


class nnUNetTrainerMedNeXt(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = int(os.environ.get("MEDNEXT_EPOCHS", 250))

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        from scala.nnunet_ext.mednext.MedNextV1 import MedNeXt
        ckpt = "outside_block" if os.environ.get("MEDNEXT_CKPT", "1") == "1" else None
        return MedNeXt(in_channels=num_input_channels, n_channels=32, n_classes=num_output_channels,
                       exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2], kernel_size=3, deep_supervision=False,
                       do_res=True, do_res_up_down=True, checkpoint_style=ckpt,
                       block_counts=[3, 4, 4, 4, 4, 4, 4, 4, 3])

    def set_deep_supervision_enabled(self, enabled: bool):
        return
