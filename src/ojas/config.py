"""Ollama auto-discovery, hardware detection, and model assignment.

On startup Ojas:
1. Pings the Ollama HTTP API at ``http://localhost:11434``.
2. Lists all locally-downloaded models.
3. Assigns the **largest** model as ``PLANNER_MODEL``.
4. Assigns a **Qwen Coder** variant as ``CODER_MODEL``.
5. Verifies that ``nomic-embed-text`` is available for RAG.
6. Offers to ``ollama pull`` any missing models.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

OLLAMA_BASE_URL = "http://localhost:11434"
REQUIRED_EMBED_MODEL = "nomic-embed-text"
PREFERRED_CODER_PATTERNS = ["qwen2.5-coder", "qwen-coder", "qwen2.5", "qwen"]
DEFAULT_PULL_MODELS = ["qwen2.5-coder:7b", "nomic-embed-text"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Lightweight representation of an Ollama model."""

    name: str
    size_bytes: int = 0
    parameter_size: str = ""
    family: str = ""
    quantization: str = ""

    @property
    def param_count_estimate(self) -> float:
        """Rough numeric parameter count parsed from the ``parameter_size`` string."""
        s = self.parameter_size.upper().strip()
        if not s:
            return self.size_bytes / 1e9  # fallback heuristic
        multiplier = 1.0
        if s.endswith("B"):
            s = s[:-1]
        if s.endswith("M"):
            s = s[:-1]
            multiplier = 0.001
        try:
            return float(s) * multiplier
        except ValueError:
            return self.size_bytes / 1e9


@dataclass
class OjasConfig:
    """Resolved runtime configuration for an Ojas session."""

    planner_model: str = ""
    coder_model: str = ""
    embed_model: str = REQUIRED_EMBED_MODEL
    ollama_base_url: str = OLLAMA_BASE_URL
    available_models: list[ModelInfo] = field(default_factory=list)
    workspace: str = "."

    @property
    def is_valid(self) -> bool:
        return bool(self.planner_model and self.coder_model and self.embed_model)


# ---------------------------------------------------------------------------
# Ollama interaction helpers
# ---------------------------------------------------------------------------


def _ping_ollama(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Return True if Ollama is reachable."""
    try:
        r = httpx.get(f"{base_url}/api/version", timeout=5)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _list_models(base_url: str = OLLAMA_BASE_URL) -> list[ModelInfo]:
    """Fetch all locally-available Ollama models."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=10)
        r.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return []

    models: list[ModelInfo] = []
    for entry in r.json().get("models", []):
        details = entry.get("details", {})
        models.append(
            ModelInfo(
                name=entry.get("name", entry.get("model", "")),
                size_bytes=entry.get("size", 0),
                parameter_size=details.get("parameter_size", ""),
                family=details.get("family", ""),
                quantization=details.get("quantization_level", ""),
            )
        )
    return models


def _pull_model(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """Pull a model via the Ollama API.  Returns True on success."""
    console.print(f"  Pulling [bold cyan]{model_name}[/] ... ", end="")
    try:
        with httpx.stream(
            "POST",
            f"{base_url}/api/pull",
            json={"name": model_name},
            timeout=None,
        ) as resp:
            for _ in resp.iter_lines():
                pass  # stream until done
        console.print("[green]done[/]")
        return True
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]failed ({exc})[/]")
        return False


# ---------------------------------------------------------------------------
# Model assignment logic
# ---------------------------------------------------------------------------


def _find_coder(
    models: list[ModelInfo],
    max_size_bytes: int | None = None,
) -> str | None:
    """Find the best Qwen coder variant from *models* that fits in RAM."""
    names_lower = {m.name.lower(): m for m in models}
    for pattern in PREFERRED_CODER_PATTERNS:
        for lower_name, model_info in names_lower.items():
            if pattern in lower_name:
                if max_size_bytes is not None and model_info.size_bytes * 1.15 > max_size_bytes:
                    continue  # too large for available RAM
                return model_info.name
    return None


def _find_embed(models: list[ModelInfo]) -> str | None:
    """Find nomic-embed-text (or variant)."""
    for m in models:
        if "nomic-embed-text" in m.name.lower():
            return m.name
    return None


def _get_available_ram_bytes() -> int | None:
    """Return available system RAM in bytes, or None if unknown."""
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.virtual_memory().available
    except ImportError:
        pass
    # Fallback: try Windows-specific approach via os.
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys
    except Exception:  # noqa: BLE001
        return None


def _find_planner(
    models: list[ModelInfo],
    exclude: str | None = None,
    max_size_bytes: int | None = None,
) -> str | None:
    """Return the name of the largest non-embedding model that fits in RAM.

    Parameters
    ----------
    models:
        All available Ollama models.
    exclude:
        Model name to exclude (typically the coder model).
    max_size_bytes:
        If provided, models whose ``size_bytes`` exceeds this value are
        skipped.  This prevents selecting a model that the system cannot
        actually load.
    """
    candidates = [
        m
        for m in models
        if "embed" not in m.name.lower() and m.name != exclude
    ]
    if max_size_bytes is not None:
        # Filter out models too large for available RAM.
        # Use a 1.15x multiplier to account for Ollama runtime overhead.
        fitting = [m for m in candidates if m.size_bytes * 1.15 <= max_size_bytes]
        if fitting:
            candidates = fitting
        else:
            # None of the non-coder models fit -- caller will fall back.
            return None
    if not candidates:
        return None
    candidates.sort(key=lambda m: m.param_count_estimate, reverse=True)
    return candidates[0].name


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def auto_configure(workspace: str = ".") -> OjasConfig:
    """Run the full auto-boot sequence and return a resolved config.

    This is called once at startup by the CLI.
    """
    cfg = OjasConfig(workspace=workspace)

    # -- Step 1: Ping Ollama -------------------------------------------------
    console.print()
    console.print(
        Panel(
            "[bold]Ojas Auto-Boot Sequence[/]",
            border_style="bright_cyan",
            expand=False,
        )
    )

    if not _ping_ollama(cfg.ollama_base_url):
        console.print(
            "[bold red]ERROR:[/] Cannot reach Ollama at "
            f"[cyan]{cfg.ollama_base_url}[/].\n"
            "Make sure Ollama is running (`ollama serve`)."
        )
        sys.exit(1)
    console.print("  [green]Ollama is online.[/]")

    # -- Step 2: Discover models ---------------------------------------------
    cfg.available_models = _list_models(cfg.ollama_base_url)
    if not cfg.available_models:
        console.print(
            "[bold red]ERROR:[/] No models found. "
            "Run `ollama pull <model>` first."
        )
        sys.exit(1)

    # -- Step 3: Detect available RAM (used for steps 4+5) --------------------
    available_ram = _get_available_ram_bytes()
    if available_ram is not None:
        ram_gb = available_ram / (1024 ** 3)
        console.print(f"  Available RAM: [cyan]{ram_gb:.1f} GiB[/]")

    # -- Step 4: Assign coder (memory-aware) ---------------------------------
    coder = _find_coder(cfg.available_models, max_size_bytes=available_ram)
    if coder is None:
        # No Qwen coder fits -- try to find ANY non-embed model that fits.
        coder = _find_planner(
            cfg.available_models,
            exclude=None,
            max_size_bytes=available_ram,
        )
        if coder:
            console.print(
                f"  [yellow]No Qwen coder fits in RAM. "
                f"Using {coder} for code generation.[/]"
            )
        else:
            console.print(
                "  [yellow]No coder model found. "
                "Pulling qwen2.5-coder:7b ...[/]"
            )
            if _pull_model("qwen2.5-coder:7b", cfg.ollama_base_url):
                cfg.available_models = _list_models(cfg.ollama_base_url)
                coder = _find_coder(cfg.available_models, max_size_bytes=available_ram)
    cfg.coder_model = coder or ""

    # -- Step 5: Assign planner (memory-aware) --------------------------------
    planner = _find_planner(
        cfg.available_models,
        exclude=cfg.coder_model,
        max_size_bytes=available_ram,
    )
    if planner is None:
        # No separate planner fits -- use coder for both roles.
        planner = cfg.coder_model
        console.print(
            "  [yellow]No separate planner model fits in RAM. "
            "Using coder model for both roles.[/]"
        )
    cfg.planner_model = planner or ""

    # -- Step 6: Verify embedding model -------------------------------------
    embed = _find_embed(cfg.available_models)
    if embed is None:
        console.print(
            "  [yellow]nomic-embed-text not found. Pulling ...[/]"
        )
        if _pull_model(REQUIRED_EMBED_MODEL, cfg.ollama_base_url):
            cfg.available_models = _list_models(cfg.ollama_base_url)
            embed = _find_embed(cfg.available_models)
    cfg.embed_model = embed or REQUIRED_EMBED_MODEL

    # -- Step 7: Report ------------------------------------------------------
    if not cfg.is_valid:
        console.print(
            "[bold red]ERROR:[/] Could not resolve all required models.\n"
            "Ensure you have at least one general model and "
            "one Qwen coder model downloaded."
        )
        sys.exit(1)

    _print_config_summary(cfg)
    return cfg


def _print_config_summary(cfg: OjasConfig) -> None:
    """Render a summary table of the resolved configuration."""
    table = Table(
        title="Model Assignment",
        show_header=True,
        header_style="bold bright_cyan",
        border_style="dim",
        expand=False,
    )
    table.add_column("Role", style="bold")
    table.add_column("Model", style="cyan")
    table.add_row("Planner (reasoning)", cfg.planner_model)
    table.add_row("Coder (generation)", cfg.coder_model)
    table.add_row("Embeddings (RAG)", cfg.embed_model)
    console.print(table)
    console.print(
        f"  Workspace: [bold]{cfg.workspace}[/]\n"
    )
