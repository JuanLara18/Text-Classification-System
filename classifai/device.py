"""
Device detection and management for classifai.

Automatically selects the best available compute device and exposes
it so every component (embeddings, clustering, LLM inference) uses
the same device without manual configuration.

Priority: CUDA > MPS (Apple Silicon) > CPU
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    device: str                     # "cuda", "cuda:0", "mps", "cpu"
    backend: str                    # "cuda", "mps", "cpu"
    name: str                       # Human-readable name
    vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    cuml_available: bool = False    # GPU-accelerated UMAP / HDBSCAN
    torch_available: bool = False

    @property
    def has_gpu(self) -> bool:
        return self.backend in ("cuda", "mps")

    def __str__(self) -> str:
        parts = [self.name]
        if self.vram_gb:
            parts.append(f"{self.vram_gb:.1f} GB VRAM")
        if self.cuda_version:
            parts.append(f"CUDA {self.cuda_version}")
        if self.cuml_available:
            parts.append("cuML available")
        return "  ".join(parts)


def detect() -> DeviceInfo:
    """
    Detect the best available compute device.

    Returns a DeviceInfo with all relevant metadata. Never raises —
    always falls back to CPU gracefully.
    """
    torch_available = False

    # ── CUDA (NVIDIA) ─────────────────────────────────────────────────────
    try:
        import torch
        torch_available = True

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            vram = props.total_memory / 1e9

            # Check cuML (RAPIDS — GPU-accelerated UMAP / HDBSCAN)
            cuml_ok = False
            try:
                import cuml  # noqa: F401
                cuml_ok = True
            except ImportError:
                pass

            cuda_ver = torch.version.cuda or "unknown"

            return DeviceInfo(
                device=f"cuda:{idx}",
                backend="cuda",
                name=props.name,
                vram_gb=round(vram, 1),
                cuda_version=cuda_ver,
                cuml_available=cuml_ok,
                torch_available=True,
            )
    except ImportError:
        pass

    # ── MPS (Apple Silicon) ───────────────────────────────────────────────
    try:
        import torch
        torch_available = True

        if torch.backends.mps.is_available():
            return DeviceInfo(
                device="mps",
                backend="mps",
                name="Apple Silicon (MPS)",
                torch_available=True,
            )
    except (ImportError, AttributeError):
        pass

    # ── CPU fallback ──────────────────────────────────────────────────────
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        name = f"CPU  ({ram_gb:.0f} GB RAM)"
    except ImportError:
        name = "CPU"

    return DeviceInfo(
        device="cpu",
        backend="cpu",
        name=name,
        torch_available=torch_available,
    )


# Module-level singleton — detected once at import time
_device_info: Optional[DeviceInfo] = None


def get() -> DeviceInfo:
    """Return the cached DeviceInfo, detecting on first call."""
    global _device_info
    if _device_info is None:
        _device_info = detect()
    return _device_info


def device_str() -> str:
    """Shorthand — returns 'cuda:0', 'mps', or 'cpu'."""
    return get().device


def sentence_transformer_device() -> str:
    """
    Device string for SentenceTransformer constructor.

    SentenceTransformer accepts 'cuda', 'mps', or 'cpu' (no index suffix).
    """
    info = get()
    if info.backend == "cuda":
        return "cuda"
    return info.backend


def print_banner() -> None:
    """
    Print a startup banner showing the detected device.
    Called once at pipeline startup so the user always knows
    what hardware is being used.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        info = get()

        if info.has_gpu:
            color = "green"
            icon = "[green]GPU[/green]"
        else:
            color = "yellow"
            icon = "[yellow]CPU[/yellow]"

        lines = [
            f"  Device  {icon}  [bold]{info.name}[/bold]",
        ]
        if info.vram_gb:
            lines.append(f"  VRAM    {info.vram_gb:.1f} GB")
        if info.cuda_version:
            lines.append(f"  CUDA    {info.cuda_version}")

        lines.append("")
        lines.append(
            "  [dim]Embeddings will run on[/dim] "
            + ("[green]GPU[/green]" if info.has_gpu else "[yellow]CPU[/yellow]")
        )
        if info.cuml_available:
            lines.append("  [dim]UMAP/HDBSCAN will run on[/dim] [green]GPU (cuML)[/green]")
        else:
            lines.append("  [dim]UMAP/HDBSCAN will run on[/dim] [yellow]CPU[/yellow]")
            if info.has_gpu:
                lines.append(
                    "  [dim]  → install cuML for GPU clustering: "
                    "https://docs.rapids.ai/install[/dim]"
                )

        console.print(
            Panel(
                "\n".join(lines),
                title="[bold]classifai[/bold]",
                border_style=color,
                padding=(0, 1),
            )
        )
    except ImportError:
        # rich not available — plain print
        info = get()
        print(f"[classifai] Device: {info}")
