"""Checkpoint-compatible SPARC reference network and sequential completion.

Scientific operations are extracted from the validated reference implementation.
Historical class/parameter names are retained for strict checkpoint loading.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
KC = 2
PHASE_COUNT = 5
COARSE_ALIGNMENT = 40
KIDX = np.arange(0, 102, 4, dtype=np.int64)

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class SliceUNet(nn.Module):
    def __init__(self, in_channels: int = 2 * (2 * KC + 1), base: int = 48):
        super().__init__()
        self.d1 = ConvBlock(in_channels, base)
        self.d2 = ConvBlock(base, base * 2)
        self.d3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.c3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.c2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.c1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        e1 = self.d1(value)
        e2 = self.d2(self.pool(e1))
        e3 = self.d3(self.pool(e2))
        middle = self.mid(self.pool(e3))
        decoded = self.c3(torch.cat([self.u3(middle), e3], dim=1))
        decoded = self.c2(torch.cat([self.u2(decoded), e2], dim=1))
        decoded = self.c1(torch.cat([self.u1(decoded), e1], dim=1))
        return self.out(decoded)

class Native150PolyphaseUNet(nn.Module):
    """Shared 30 MHz backbone over all five native temporal phases."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = SliceUNet()
        self.temporal_refiner = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 11), padding=(0, 5)),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=(1, 11), padding=(0, 5)),
        )
        nn.init.zeros_(self.temporal_refiner[-1].weight)
        nn.init.zeros_(self.temporal_refiner[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, original_time = value.shape
        pad_time = (-original_time) % COARSE_ALIGNMENT
        if pad_time:
            value = F.pad(value, (0, pad_time, 0, 0), mode="reflect")
        padded_time = value.shape[-1]
        coarse_time = padded_time // PHASE_COUNT
        phases = (
            value.reshape(
                batch,
                channels,
                height,
                coarse_time,
                PHASE_COUNT,
            )
            .permute(0, 4, 1, 2, 3)
            .reshape(
                batch * PHASE_COUNT,
                channels,
                height,
                coarse_time,
            )
        )
        phase_output = self.backbone(phases)
        interleaved = (
            phase_output.reshape(
                batch,
                PHASE_COUNT,
                1,
                height,
                coarse_time,
            )
            .permute(0, 2, 3, 4, 1)
            .reshape(batch, 1, height, padded_time)
        )
        output = interleaved + self.temporal_refiner(interleaved)
        return output[..., :original_time]

    def initialize_backbone(self, checkpoint_path: Path) -> dict:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.backbone.load_state_dict(payload["model"], strict=True)
        return payload


def _run_model(
    model: nn.Module,
    inputs: torch.Tensor,
    use_checkpoint: bool,
) -> torch.Tensor:
    if use_checkpoint and torch.is_grad_enabled():
        return checkpoint(model, inputs, use_reentrant=False)
    return model(inputs)


def fill_axis_batched(
    model: nn.Module,
    field: torch.Tensor,
    indices: torch.Tensor,
    column_batch: int = 8,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    spatial_size, column_count, time_size = field.shape
    mask = torch.zeros(spatial_size, time_size, device=field.device)
    mask[indices] = 1.0
    outputs = []
    for start in range(0, column_count, column_batch):
        stop = min(start + column_batch, column_count)
        columns = torch.arange(start, stop, device=field.device)
        channels = []
        for delta in range(-KC, KC + 1):
            context_indices = torch.clamp(
                columns + delta, 0, column_count - 1
            )
            context = field[:, context_indices].permute(1, 0, 2)
            channels.extend(
                [
                    context * mask.unsqueeze(0),
                    mask.unsqueeze(0).expand(stop - start, -1, -1),
                ]
            )
        inputs = torch.stack(channels, dim=1)
        padded_height = (spatial_size + 15) // 16 * 16
        if padded_height != spatial_size:
            inputs = F.pad(
                inputs,
                (0, 0, 0, padded_height - spatial_size),
                mode="reflect",
            )
        prediction = _run_model(model, inputs, use_checkpoint)
        outputs.append(
            prediction[:, 0, :spatial_size].permute(1, 0, 2)
        )
    output = torch.cat(outputs, dim=1)
    known = mask.bool().unsqueeze(1).expand(-1, column_count, -1)
    return torch.where(known, field, output)


def cascade_tensor_native150(
    model: nn.Module,
    active: torch.Tensor,
    column_batch: int = 8,
    use_checkpoint: bool = False,
    known_indices: np.ndarray | None = None,
) -> torch.Tensor:
    indices_np = (
        KIDX
        if known_indices is None
        else np.asarray(known_indices, dtype=np.int64)
    )
    indices = torch.as_tensor(
        indices_np, dtype=torch.long, device=active.device
    )
    spatial_size = active.shape[0]
    known = active[indices][:, indices]
    first = torch.zeros(
        spatial_size,
        len(indices),
        active.shape[-1],
        dtype=active.dtype,
        device=active.device,
    )
    first[indices] = known
    first = fill_axis_batched(
        model,
        first,
        indices,
        column_batch=column_batch,
        use_checkpoint=use_checkpoint,
    )
    second = torch.zeros(
        spatial_size,
        spatial_size,
        active.shape[-1],
        dtype=active.dtype,
        device=active.device,
    )
    second[:, indices] = first
    second = fill_axis_batched(
        model,
        second.permute(1, 0, 2),
        indices,
        column_batch=column_batch,
        use_checkpoint=use_checkpoint,
    ).permute(1, 0, 2)
    second = second.clone()
    second[indices[:, None], indices[None, :]] = known
    return second
