"""Export deterministic locomotion policies with their history encoders.

Inputs are flattened, preprocessed actor observations ordered oldest to newest.
Outputs are raw actions; action clipping, scaling and offsets remain external.
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import CTSActorCritic, DreamWaQActorCritic, HIMActorCritic, HIMAMPActorCritic


class _HIMPolicyExporter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.actor = copy.deepcopy(model.actor)
        self.num_one_step_obs = model.num_one_step_obs
        self.num_actor_obs = model.num_actor_obs
        self.encoder = copy.deepcopy(model.estimator.encoder)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        parts = self.encoder(obs_history)
        velocity = parts[:, :3]
        latent = F.normalize(parts[:, 3:], p=2.0, dim=-1)
        current_obs = obs_history[:, -self.num_one_step_obs:]
        return self.actor(torch.cat((current_obs, velocity, latent), dim=-1))


class _DreamWaQPolicyExporter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.actor = copy.deepcopy(model.actor)
        self.num_one_step_obs = model.num_one_step_obs
        self.num_actor_obs = model.num_history_obs + model.num_one_step_obs
        self.encoder = copy.deepcopy(model.vae.encoder)
        self.mean_velocity = copy.deepcopy(model.vae.encode_mean_vel)
        self.mean_latent = copy.deepcopy(model.vae.encode_mean_latent)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(obs_history[:, :-self.num_one_step_obs])
        velocity = self.mean_velocity(encoded)
        latent = self.mean_latent(encoded)
        current_obs = obs_history[:, -self.num_one_step_obs:]
        return self.actor(torch.cat((velocity, latent, current_obs), dim=-1))


class _CTSPolicyExporter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.actor = copy.deepcopy(model.actor)
        self.num_one_step_obs = model.num_one_step_obs
        self.num_actor_obs = model.num_actor_obs
        self.student_encoder = copy.deepcopy(model.student_encoder)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        latent = self.student_encoder(obs_history[:, :-self.num_one_step_obs])
        current_obs = obs_history[:, -self.num_one_step_obs:]
        return self.actor(torch.cat((latent, current_obs), dim=-1))


def _make_policy_exporter(model: nn.Module) -> nn.Module:
    if isinstance(model, (HIMActorCritic, HIMAMPActorCritic)):
        exporter = _HIMPolicyExporter(model)
    elif isinstance(model, DreamWaQActorCritic):
        exporter = _DreamWaQPolicyExporter(model)
    elif isinstance(model, CTSActorCritic):
        exporter = _CTSPolicyExporter(model)
    else:
        raise TypeError(f"Unsupported policy type for export: {type(model).__name__}")
    return exporter.cpu().eval()


def export_policy_as_jit(
    model: nn.Module,
    path: str | Path,
    filename: str = "policy.pt",
) -> None:
    """Export encoder and actor as one TorchScript model, without moving model."""
    exporter = _make_policy_exporter(model)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.jit.script(exporter).save(str(path / filename))
    print(f"[INFO] Exported TorchScript policy to: {path / filename}")


def export_policy_as_onnx(
    model: nn.Module,
    path: str | Path,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export [batch, history * frame] -> [batch, actions] with a dynamic batch."""
    exporter = _make_policy_exporter(model)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    obs_history = torch.zeros(1, exporter.num_actor_obs)
    torch.onnx.export(
        exporter,
        obs_history,
        str(path / filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=["obs_history"],
        output_names=["actions"],
        dynamic_axes={"obs_history": {0: "batch"}, "actions": {0: "batch"}},
        dynamo=False,
    )
    print(f"[INFO] Exported ONNX policy to: {path / filename}")
