"""
ACE-Step Custom Setup (v11 — hardware autodetect + installation profiles)
==============================================================
Performs a clean install of ACE-Step v1.5 from GitHub, detects GPU
hardware, recommends an installation profile, downloads only the
models needed, and applies cumulative VRAM + GUI patches.

Can also be run in --patch-only mode to re-apply patches to an
existing installation without cloning or downloading models.

VRAM / SCORING PATCHES (file replacements):
    - llm_inference.py       : HF scoring model always loads to CPU first;
                               Blackwell auto-detect + torch.compile disable;
                               VRAM leak fix in unload()
    - lm_score.py            : Scoring model swaps GPU only during forward pass
    - scoring.py             : DiT offload before PMI, serialised scoring lock,
                               tensor cleanup after each scored batch,
                               + lyric_token_idss typo fix
    - lyric_score.py         : Explicit VRAM cleanup after attention pass
    - model_runner.py        : nano-vllm ModelRunner.exit() properly frees
                               model weights, KV cache, sampler, pinned buffers

WINDOWS / BLACKWELL DEFAULTS:
    - generation_defaults.py : Compile model defaults to OFF on Windows
                               (torch.compile requires Triton, unavailable)
    - service_init.py        : Tier-change handler respects Windows compile default

GUI DISPLAY BUG FIXES (text-level patches):
    Fix A: generation_progress.py  — Disable progressive yield inside loop
    Fix B: batch_management.py     — Remove Phase 1/2 audio overwrite block
    Fix C: acestep_v15_pipeline.py — Normalise output_dir path separators;
                                     add allowed_paths to demo.launch()
    Fix D: result.py + pipeline    — Batch samples 3 & 4 visible on first load
                                     (respects default_batch_size from .env)

REPLACED IN v4 (previously Fix E in v3 — caused GUI breakage):
    Fix E: pipeline UI file        — Key / Time Signature / Vocal Language
                                     fields: only patches explicit visible=False
                                     on label-matched components.
                                     REMOVED: nuclear CSS injection
                                     REMOVED: blind event commenting
                                     REMOVED: generic 'key' variable matching

    batch_navigation.py            — Full file replacement with forward-slash
                                     audio path normalisation (Windows / Gradio 6)

TOOLTIP / UX IMPROVEMENTS (v10):
    Fix H: interfaces/__init__.py  — Tooltip CSS (max-height + scrollable) and JS
                                     (flip above viewport bottom) so long tooltips
                                     are always readable.
    Fix I: generation_advanced_dit_controls.py — Add info= tooltips to CFG Interval
                                     Start/End sliders (were missing entirely).

    en.json                        — Full file replacement with practical, user-facing
                                     hover descriptions for every GUI setting. Explains
                                     what each parameter does, when to raise/lower it,
                                     and warns about edge cases (e.g. high CFG values).

    llm_inference.py               — Suppress verbose Triton/traceback on Windows
                                     startup when vLLM falls back to PyTorch.

HARDWARE AUTODETECT + INSTALLATION PROFILES (v11):
    Detects GPU (NVIDIA/AMD/Intel) and system RAM before cloning.
    Presents 4 profiles (Express/Balanced/Quality/Studio) with a
    recommendation based on detected VRAM. Only downloads models
    needed for the chosen profile. Configures .env with profile-
    specific defaults (model paths, batch size).
"""

import os
import re
import sys
import shutil
import subprocess
import argparse
import tempfile
import time
import binascii
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}\u2713{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}!{RESET}  {msg}")
def err(msg):   print(f"  {RED}\u2717{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}\u00bb{RESET}  {msg}")
def header(msg):print(f"\n{BOLD}{CYAN}{msg}{RESET}")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
PATCHES_DIR = SCRIPT_DIR / "patches"
CONFIG_DIR  = SCRIPT_DIR / "config"

# ── Package integrity check ──────────────────────────────────────────────────
def _crc32_file(path: Path) -> str:
    """Compute CRC32 hex string for a file."""
    crc = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            crc = binascii.crc32(chunk, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def verify_integrity():
    """Verify CRC32 checksums of all package files against checksums.txt.
    Returns True if all files pass or if checksums.txt is missing (dev mode).
    Prints warnings for any mismatches."""
    checksums_file = SCRIPT_DIR / "checksums.txt"
    if not checksums_file.exists():
        return True  # No manifest — skip check (development mode)

    info("Verifying package integrity...")
    expected = {}
    for line in checksums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            expected[parts[1]] = parts[0]

    all_ok = True
    for rel_path, exp_crc in expected.items():
        full_path = SCRIPT_DIR / rel_path
        if not full_path.exists():
            warn(f"  MISSING: {rel_path}")
            all_ok = False
            continue
        actual_crc = _crc32_file(full_path)
        if actual_crc != exp_crc:
            warn(f"  MODIFIED: {rel_path}  (expected {exp_crc}, got {actual_crc})")
            all_ok = False

    if all_ok:
        ok("Package integrity verified — all files match.")
    else:
        print()
        warn("WARNING: One or more package files have been modified!")
        warn("This may indicate the package was tampered with or corrupted.")
        warn("Re-download the original package if you did not make these changes.")
        print()
        resp = input("  Continue anyway? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            err("Installation aborted by user.")
            sys.exit(1)

    return all_ok


# ── Models to download ────────────────────────────────────────────────────────
# The default `acestep-download` without --model gets turbo + 1.7B LM.
# We also want the SFT base and all three 5Hz LM variants.
MODELS = [
    "acestep-v15-base",
    "acestep-v15-sft",
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-4B",
]

# ── Installation profiles ────────────────────────────────────────────────────
PROFILES = [
    {
        "name": "Express",
        "description": "Fast generation, low VRAM",
        "detail": "turbo + 0.6B LM  | ~5 GB  | Fast, basic Think mode",
        "target_vram": "6-8 GB",
        "dit_model": "acestep-v15-turbo",
        "lm_model": "acestep-5Hz-lm-0.6B",
        "batch_size": 1,
        "dit_steps": 8,
        "est_vram": "~5 GB",
        "est_vram_gb": 5,
        "min_vram_gb": 0,
        "max_vram_gb": 8,
        "models_to_download": [
            "acestep-5Hz-lm-0.6B",
        ],
    },
    {
        "name": "Balanced",
        "description": "Good quality, moderate VRAM",
        "detail": "SFT + 1.7B LM   | ~8 GB  | Full features, good quality",
        "target_vram": "10-12 GB",
        "dit_model": "acestep-v15-sft",
        "lm_model": "acestep-5Hz-lm-1.7B",
        "batch_size": 2,
        "dit_steps": 50,
        "est_vram": "~8 GB",
        "est_vram_gb": 8,
        "min_vram_gb": 8,
        "max_vram_gb": 12,
        "models_to_download": [
            "acestep-v15-sft",
            "acestep-5Hz-lm-1.7B",
        ],
    },
    {
        "name": "Quality",
        "description": "Best LM, full features",
        "detail": "SFT + 4B LM     | ~12 GB | Best LM, full features",
        "target_vram": "16 GB",
        "dit_model": "acestep-v15-sft",
        "lm_model": "acestep-5Hz-lm-4B",
        "batch_size": 2,
        "dit_steps": 50,
        "est_vram": "~12 GB",
        "est_vram_gb": 12,
        "min_vram_gb": 12,
        "max_vram_gb": 20,
        "models_to_download": [
            "acestep-v15-sft",
            "acestep-5Hz-lm-4B",
        ],
    },
    {
        "name": "Studio",
        "description": "Maximum quality, batch 4",
        "detail": "SFT + 4B LM     | ~14 GB | Best quality, batch 4",
        "target_vram": "24+ GB",
        "dit_model": "acestep-v15-sft",
        "lm_model": "acestep-5Hz-lm-4B",
        "batch_size": 4,
        "dit_steps": 50,
        "est_vram": "~14 GB",
        "est_vram_gb": 14,
        "min_vram_gb": 20,
        "max_vram_gb": 999,
        "models_to_download": [
            "acestep-v15-sft",
            "acestep-5Hz-lm-4B",
        ],
    },
]


# ── File-replacement patch map ────────────────────────────────────────────────
def get_patch_map(root: Path) -> dict:
    return {
        "llm_inference.py":       root / "acestep" / "llm_inference.py",
        "lm_score.py":            root / "acestep" / "core" / "scoring" / "lm_score.py",
        "scoring.py":             root / "acestep" / "ui" / "gradio" / "events" / "results" / "scoring.py",
        "lyric_score.py":         root / "acestep" / "core" / "generation" / "handler" / "lyric_score.py",
        "batch_navigation.py":    root / "acestep" / "ui" / "gradio" / "events" / "results" / "batch_navigation.py",
        "model_runner.py":        root / "acestep" / "third_parts" / "nano-vllm" / "nanovllm" / "engine" / "model_runner.py",
        "generation_defaults.py": root / "acestep" / "ui" / "gradio" / "interfaces" / "generation_defaults.py",
        "service_init.py":        root / "acestep" / "ui" / "gradio" / "events" / "generation" / "service_init.py",
        "en.json":                root / "acestep" / "ui" / "gradio" / "i18n" / "en.json",
    }

PATCH_SENTINELS = {
    "llm_inference.py":       "will move to GPU only during forward pass",
    "lm_score.py":            "regardless of the\n    # offload_to_cpu flag",
    "scoring.py":             "_SCORING_LOCK = threading.Lock()",
    "lyric_score.py":         "Explicitly free all large intermediate CUDA tensors",
    "batch_navigation.py":    "GUI_FIX_V2: batch_navigation patched",
    "model_runner.py":        "Release model weights, KV cache, and sampler from GPU",
    "generation_defaults.py": "torch.compile needs a working Triton backend",
    "service_init.py":        "_compile_default",
    "en.json":                "cfg_interval_start_info",
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def backup_if_exists(path: Path):
    if path.exists():
        backup = path.with_suffix(path.suffix + ".backup")
        if not backup.exists():
            shutil.copy2(path, backup)
            info(f"Backed up: {path.name} -> {path.name}.backup")

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def find_by_content(root: Path, marker: str) -> Path | None:
    """Return the first .py under acestep that contains marker."""
    for root_dir, dirs, files in os.walk(root / "acestep"):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in files:
            if fname.endswith(".py"):
                fp = Path(root_dir) / fname
                try:
                    if marker in read_file(fp):
                        return fp
                except Exception:
                    pass
    return None

def find_all_by_content(root: Path, marker: str) -> list:
    """Return all .py files under acestep that contain marker."""
    results = []
    for root_dir, dirs, files in os.walk(root / "acestep"):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in files:
            if fname.endswith(".py"):
                fp = Path(root_dir) / fname
                try:
                    if marker in read_file(fp):
                        results.append(fp)
                except Exception:
                    pass
    return results

def run_cmd(cmd, cwd=None, check=True, label=""):
    """Run a subprocess command, printing output in real time."""
    info(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        warn(f"{label or cmd[0]}: returned exit code {result.returncode}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Hardware detection (runs before clone — zero Python deps)
# ═══════════════════════════════════════════════════════════════════════════

def _detect_gpu_nvidia_smi() -> list:
    """Try nvidia-smi --query-gpu for NVIDIA GPUs. Returns list of (name, vram_mb)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    gpus.append((parts[0], int(parts[1])))
                except ValueError:
                    pass
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _detect_gpu_dxdiag() -> list:
    """Use dxdiag /t to detect any GPU (NVIDIA/AMD/Intel). Returns list of (name, vram_mb)."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"_acestep_dxdiag_{os.getpid()}.txt")
    try:
        # dxdiag /t writes output and exits
        subprocess.run(
            ["dxdiag", "/t", tmp_path],
            capture_output=True, timeout=30,
        )
        # dxdiag may need a moment to write the file
        for _ in range(10):
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100:
                break
            time.sleep(0.5)

        if not os.path.exists(tmp_path):
            return []

        # dxdiag output encoding varies by Windows version (UTF-8 or UTF-16-LE)
        content = None
        for enc in ("utf-8", "utf-16-le", "utf-16", "cp1252"):
            try:
                candidate = open(tmp_path, encoding=enc, errors="replace").read()
                if "card name" in candidate.lower():
                    content = candidate
                    break
            except Exception:
                continue
        if content is None:
            return []

        gpus = []
        current_name = None
        for line in content.splitlines():
            line = line.strip()
            # Card name line: "Card name: NVIDIA GeForce RTX 5090"
            if line.lower().startswith("card name:"):
                current_name = line.split(":", 1)[1].strip()
            # Dedicated Memory line: "Dedicated Memory: 32384 MB"
            # Some versions use "Display Memory" instead
            elif (line.lower().startswith("dedicated memory:") or
                  line.lower().startswith("display memory (vram):")) and current_name:
                mem_str = line.split(":", 1)[1].strip()
                mem_str = mem_str.replace("MB", "").replace("mb", "").strip()
                try:
                    vram_mb = int(mem_str)
                    gpus.append((current_name, vram_mb))
                except ValueError:
                    pass
                current_name = None
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _detect_gpu_wmi() -> list:
    """Fallback: PowerShell Win32_VideoController. Works for all GPU vendors.
    WARNING: AdapterRAM is uint32 and caps at ~4 GB for cards with more VRAM."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,AdapterRAM | "
             "ForEach-Object { $_.Name + '|' + $_.AdapterRAM }"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2 and parts[1].strip():
                name = parts[0].strip()
                # Skip generic display adapters (e.g. Microsoft Basic Display)
                if "basic" in name.lower() or "microsoft" in name.lower():
                    continue
                try:
                    vram_bytes = int(parts[1].strip())
                    vram_mb = vram_bytes // (1024 * 1024)
                    if vram_mb > 0:
                        gpus.append((name, vram_mb))
                except ValueError:
                    pass
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _detect_system_ram() -> float | None:
    """Detect total system RAM in GB. Uses wmic, then PowerShell fallback."""
    # Try wmic first
    try:
        result = subprocess.run(
            ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory",
             "/format:value"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "TotalPhysicalMemory" in line:
                val = line.split("=", 1)[1].strip()
                if val:
                    return int(val) / (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
        pass

    # PowerShell fallback (wmic is deprecated on Win11 24H2+)
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
            capture_output=True, text=True, timeout=10,
        )
        val = result.stdout.strip()
        if val:
            return int(val) / (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
        pass

    return None


def detect_hardware() -> dict:
    """Detect GPU and system RAM using only system utilities (no Python deps).

    Returns dict with: gpu_name, vram_mb, vram_gb, system_ram_gb,
    all_gpus (list), gpu_vendor, detection_method.
    """
    header("Hardware Detection")

    hardware = {
        "gpu_name": None,
        "vram_mb": None,
        "vram_gb": None,
        "system_ram_gb": None,
        "all_gpus": [],
        "gpu_vendor": None,
        "detection_method": None,
    }

    # ── GPU detection: three-tier fallback ──────────────────────────
    gpus = []
    method = None

    # Tier 1: nvidia-smi (NVIDIA only, most accurate)
    info("Checking for NVIDIA GPU (nvidia-smi)...")
    gpus = _detect_gpu_nvidia_smi()
    if gpus:
        method = "nvidia-smi"

    # Tier 2: dxdiag (universal, accurate VRAM)
    if not gpus:
        info("Checking for GPU via DirectX diagnostics (dxdiag)...")
        gpus = _detect_gpu_dxdiag()
        if gpus:
            method = "dxdiag"

    # Tier 3: WMI (universal, but VRAM may cap at 4 GB)
    if not gpus:
        info("Checking for GPU via Windows Management (WMI)...")
        gpus = _detect_gpu_wmi()
        if gpus:
            method = "WMI (VRAM may be underreported for cards >4 GB)"

    if gpus:
        hardware["all_gpus"] = gpus
        hardware["detection_method"] = method
        # Pick GPU with most VRAM
        best = max(gpus, key=lambda g: g[1])
        hardware["gpu_name"] = best[0]
        hardware["vram_mb"] = best[1]
        hardware["vram_gb"] = round(best[1] / 1024, 1)

        # Determine vendor
        name_lower = best[0].lower()
        if "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower or "gtx" in name_lower:
            hardware["gpu_vendor"] = "NVIDIA"
        elif "amd" in name_lower or "radeon" in name_lower:
            hardware["gpu_vendor"] = "AMD"
        elif "intel" in name_lower or "arc" in name_lower:
            hardware["gpu_vendor"] = "Intel"
        else:
            hardware["gpu_vendor"] = "Unknown"

        ok(f"GPU: {hardware['gpu_name']}")
        ok(f"VRAM: {hardware['vram_gb']} GB  (detected via {method})")
        if len(gpus) > 1:
            info(f"  ({len(gpus)} GPUs detected — using the one with most VRAM)")
            for i, (name, mb) in enumerate(gpus):
                info(f"    GPU {i}: {name} — {round(mb/1024, 1)} GB")
    else:
        warn("No discrete GPU detected")
        warn("ACE-Step requires a GPU for music generation.")
        info("If you have a GPU, make sure drivers are installed.")

    # ── System RAM ──────────────────────────────────────────────────
    info("Checking system memory...")
    hardware["system_ram_gb"] = _detect_system_ram()
    if hardware["system_ram_gb"]:
        ok(f"System RAM: {hardware['system_ram_gb']:.0f} GB")
    else:
        warn("Could not detect system RAM")

    return hardware


def recommend_profile(hardware: dict) -> int:
    """Return the index into PROFILES that best matches detected hardware."""
    vram = hardware.get("vram_gb")
    ram = hardware.get("system_ram_gb")

    if vram is None:
        return 0  # Express — safest default when detection fails

    # Walk profiles in reverse — return the highest tier that fits
    recommended = 0
    for i, p in enumerate(PROFILES):
        if vram >= p["min_vram_gb"]:
            recommended = i

    # Cap at Balanced if system RAM < 16 GB (4B LM needs RAM for CPU offload)
    if ram is not None and ram < 16 and recommended > 1:
        recommended = 1

    return recommended


def prompt_profile_selection(hardware: dict) -> dict:
    """Display hardware info, show profile menu, return chosen profile dict."""
    rec_idx = recommend_profile(hardware)
    vram_gb = hardware.get("vram_gb")

    header("Installation Profiles")
    print()

    for i, p in enumerate(PROFILES):
        marker = "  << RECOMMENDED" if i == rec_idx else ""
        print(f"    [{i+1}] {p['name']:<12} {p['detail']}{marker}")

    print()
    if vram_gb:
        info(f"Recommended for your GPU ({vram_gb} GB): [{rec_idx+1}] {PROFILES[rec_idx]['name']}")
    else:
        info(f"Default recommendation: [{rec_idx+1}] {PROFILES[rec_idx]['name']}")
    print()

    while True:
        choice = input(f"  Select profile (1-{len(PROFILES)}, default {rec_idx+1}): ").strip()
        if choice == "":
            chosen = rec_idx
            break
        try:
            chosen = int(choice) - 1
            if 0 <= chosen < len(PROFILES):
                break
            warn(f"Please enter a number between 1 and {len(PROFILES)}")
        except ValueError:
            warn(f"Please enter a number between 1 and {len(PROFILES)}")

    profile = PROFILES[chosen]

    # Warn if chosen profile exceeds detected VRAM
    if vram_gb and profile["est_vram_gb"] > vram_gb:
        print()
        warn(f"{profile['name']} ({profile['est_vram']}) may exceed your {vram_gb} GB VRAM.")
        warn("The system will offload to CPU when needed (slower).")
        confirm = input("  Proceed anyway? (Y/N, default Y): ").strip().upper()
        if confirm in ("N", "NO"):
            info("Returning to profile selection...")
            return prompt_profile_selection(hardware)

    print()
    ok(f"Selected profile: {profile['name']} — {profile['description']}")
    return profile


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Clone the repository
# ═══════════════════════════════════════════════════════════════════════════

def clone_repo(install_dir: Path):
    header("Step 1 -- Cloning ACE-Step v1.5 from GitHub")

    if (install_dir / "acestep").exists():
        warn(f"ACE-Step already exists at: {install_dir}")
        warn("Skipping clone — will apply patches to existing installation.")
        return True

    install_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "clone", "https://github.com/ACE-Step/ACE-Step-1.5.git", str(install_dir)],
        capture_output=False,
    )
    if result.returncode != 0:
        err("git clone failed. Make sure git is installed and accessible.")
        return False

    ok(f"Cloned ACE-Step v1.5 to: {install_dir}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Set up environment + download models
# ═══════════════════════════════════════════════════════════════════════════

def setup_environment(root: Path):
    header("Step 2 -- Setting up environment (uv sync)")

    embedded_python = root / "python_embeded" / "python.exe"
    use_embedded = embedded_python.exists()

    if use_embedded:
        info("Using embedded Python — skipping uv sync")
        return "embedded"

    info("Running 'uv sync' to set up / update environment...")
    try:
        result = subprocess.run(["uv", "sync"], cwd=root, capture_output=False)
        if result.returncode == 0:
            ok("uv sync completed")
        else:
            warn("uv sync returned a non-zero exit code — continuing.")
    except FileNotFoundError:
        err("'uv' not found. Please install uv first:")
        err("  powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        return None

    return "uv"


def install_hf_xet(root: Path, python_mode: str):
    """Install hf_xet for faster Hugging Face downloads via Xet storage.
    
    IMPORTANT: Uses 'uv pip install' rather than 'uv add' to avoid modifying
    pyproject.toml and triggering a full dependency re-resolution, which can
    drop packages like flash-attn that were installed by 'uv sync'.
    """
    info("Installing hf_xet for faster model downloads...")

    embedded_python = root / "python_embeded" / "python.exe"
    venv_python = root / ".venv" / "Scripts" / "python.exe"

    try:
        if python_mode == "embedded":
            result = subprocess.run(
                [str(embedded_python), "-m", "pip", "install", "hf_xet"],
                cwd=root, capture_output=False,
            )
        else:
            result = subprocess.run(
                ["uv", "pip", "install", "--python", str(venv_python), "hf_xet"],
                cwd=root, capture_output=False,
            )

        if result.returncode == 0:
            ok("hf_xet installed — model downloads will use Xet acceleration")
        else:
            warn("hf_xet install failed — downloads will use standard HTTP (still works, just slower)")
    except Exception as e:
        warn(f"Could not install hf_xet: {e} — downloads will use standard HTTP")


def download_models(root: Path, python_mode: str, profile: dict = None):
    """Download models. If a profile is provided, only download that profile's
    models. Otherwise download ALL models (legacy / patch-only behaviour)."""
    embedded_python = root / "python_embeded" / "python.exe"

    if profile:
        models = profile["models_to_download"]
        header(f"Step 3 -- Downloading models for {profile['name']} profile")
        info(f"Models: {', '.join(models)}")
    else:
        models = MODELS
        header("Step 3 -- Downloading ALL models")
        info("This will download: base, SFT, turbo configs + 0.6B, 1.7B, 4B LM models")

    info("(Already-downloaded models will be skipped automatically)")
    print()

    # The bare `acestep-download` (no --model) always downloads turbo + 1.7B LM.
    # Run it first to ensure the base turbo config is present for all profiles.
    info("Running base downloader (turbo config + default LM)...")
    if python_mode == "embedded":
        result = subprocess.run(
            [str(embedded_python), "-m", "acestep.model_downloader"],
            cwd=root, capture_output=False,
        )
    else:
        result = subprocess.run(
            ["uv", "run", "acestep-download"],
            cwd=root, capture_output=False,
        )
    if result.returncode == 0:
        ok("Base download completed")
    else:
        warn("Base download returned non-zero — will try individual models")

    # Download each model specified by the profile
    for model in models:
        info(f"Ensuring model present: {model}")
        try:
            if python_mode == "embedded":
                result = subprocess.run(
                    [str(embedded_python), "-m", "acestep.model_downloader", "--model", model],
                    cwd=root, capture_output=False,
                )
            else:
                result = subprocess.run(
                    ["uv", "run", "acestep-download", "--model", model],
                    cwd=root, capture_output=False,
                )
            if result.returncode == 0:
                ok(f"Model ready: {model}")
            else:
                warn(f"Download may have failed for {model} (exit code {result.returncode})")
        except FileNotFoundError as e:
            warn(f"Could not run downloader for {model}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Credential prompting
# ═══════════════════════════════════════════════════════════════════════════

def prompt_credentials() -> tuple:
    """Ask the user for a web UI username and password, or let them
    disable remote sharing entirely.
    Returns (username, password) tuple, or None if sharing is disabled."""
    header("Remote Access Credentials")
    print()
    info("ACE-Step's web UI can be accessed remotely over your network.")
    info("Setting a login prevents unauthorised access when sharing.")
    print()
    print("  1) Set a username and password for remote access")
    print("  2) Disable remote sharing (local-only, no login required)")
    print()

    while True:
        choice = input("  Choose [1/2] (default: 1): ").strip()
        if choice in ("", "1"):
            break
        if choice == "2":
            print()
            ok("Remote sharing disabled — web UI will only be accessible on this machine.")
            return None
        warn("Please enter 1 or 2.")

    print()

    while True:
        username = input("  Enter a username for the web UI: ").strip()
        if username:
            break
        warn("Username cannot be empty — please try again.")

    while True:
        password = input("  Enter a password for the web UI: ").strip()
        if password:
            break
        warn("Password cannot be empty — please try again.")

    print()
    ok(f"Credentials set: {username} / {'*' * len(password)}")
    return username, password


def test_vllm_compatibility(root: Path, python_mode: str) -> bool:
    """Run a CUDA availability test.  Returns True when the PyTorch backend
    should be used instead of vLLM.  This happens when:
      - No CUDA GPU is detected at all

    On Blackwell + Windows, nano-vllm cannot use CUDA graphs (broken on sm_120)
    or torch.compile (Triton unavailable on Windows).  However, nano-vllm in
    eager mode (with torch._dynamo disabled) still outperforms the raw PyTorch
    backend (~50-55 tok/s vs ~36 tok/s).  The VRAM leak in unload() has been
    fixed, so vLLM is now the recommended backend for all CUDA GPUs."""
    header("vLLM Compatibility Check")
    info("Testing CUDA availability for vLLM backend...")
    print()

    # Write a small test script to run inside the project's venv
    test_script = root / "_vllm_compat_test.py"
    test_code = '''\
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("NO_CUDA")
        sys.exit(0)

    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name}")
    print(f"COMPUTE: sm_{major}{minor}")

    # Test basic CUDA tensor ops (sanity check)
    device = torch.device("cuda")
    x = torch.randn(4, 16, device=device)
    w = torch.randn(16, 16, device=device)
    out = torch.mm(x, w)
    torch.cuda.synchronize()

    # Report CUDA graph capability (informational, not a blocker)
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = torch.mm(x, w)
        graph.replay()
        torch.cuda.synchronize()
        print("CUDA_GRAPH_OK")
    except Exception as e:
        print(f"CUDA_GRAPH_FAIL: {e}")
        # This is OK — the patched llm_inference.py will use enforce_eager

    print("CUDA_OK")
except Exception as e:
    print(f"CUDA_FAIL: {e}")
'''
    test_script.write_text(test_code, encoding="utf-8")

    try:
        embedded_python = root / "python_embeded" / "python.exe"
        if python_mode == "embedded":
            cmd = [str(embedded_python), str(test_script)]
        else:
            cmd = ["uv", "run", "--no-project", "python", str(test_script)]
            # Try project python directly as fallback
            venv_python = root / ".venv" / "Scripts" / "python.exe"
            if venv_python.exists():
                cmd = [str(venv_python), str(test_script)]

        result = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout + result.stderr).strip()

        # Print GPU info
        for line in output.splitlines():
            if line.startswith("GPU:") or line.startswith("COMPUTE:"):
                info(f"Detected: {line}")

        if "NO_CUDA" in output:
            info("No CUDA GPU detected — PyTorch backend will be used")
            return True
        elif "CUDA_OK" in output:
            if "CUDA_GRAPH_OK" in output:
                ok("CUDA graph capture works — vLLM will use full acceleration")
            elif "CUDA_GRAPH_FAIL" in output:
                info("CUDA graph capture not available on this GPU (e.g. Blackwell)")
                info("vLLM will run in eager mode (enforce_eager + dynamo disabled)")
                ok("This is normal — vLLM eager still outperforms PyTorch backend")
            ok("vLLM backend will be used")
            return False
        elif "CUDA_FAIL" in output:
            warn("CUDA test failed — falling back to PyTorch backend")
            for line in output.splitlines():
                if "CUDA_FAIL" in line:
                    info(f"  Reason: {line.split('CUDA_FAIL: ', 1)[-1]}")
            return True
        else:
            warn("Could not determine CUDA compatibility")
            warn(f"Test output: {output[:200]}")
            print()
            answer = input("  Set LM backend to PyTorch for safety? (Y/N, default Y): ").strip().upper()
            if answer in ("N", "NO"):
                ok("LM backend will use the default (vLLM)")
                return False
            else:
                ok("LM backend will be set to PyTorch (pt)")
                return True

    except subprocess.TimeoutExpired:
        warn("GPU test timed out (30s) — defaulting to PyTorch backend")
        return True
    except Exception as e:
        warn(f"Could not run GPU test: {e}")
        print()
        answer = input("  Set LM backend to PyTorch for safety? (Y/N, default Y): ").strip().upper()
        if answer in ("N", "NO"):
            return False
        return True
    finally:
        # Clean up test script
        if test_script.exists():
            test_script.unlink()


def _upsert_env_value(content: str, key: str, value: str) -> str:
    """Update or insert a KEY=VALUE pair in .env content."""
    if re.search(rf'^{re.escape(key)}\s*=', content, re.MULTILINE):
        return re.sub(
            rf'^{re.escape(key)}\s*=.*$',
            f'{key}={value}',
            content, flags=re.MULTILINE,
        )
    else:
        return content.rstrip() + f'\n{key}={value}\n'


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Install config files
# ═══════════════════════════════════════════════════════════════════════════

def install_config(root: Path, credentials: tuple = None, force_pt_backend: bool = False, profile: dict = None):
    header("Step 4 -- Installing config files (.env + Start_Custom.bat)")

    # .env
    src_env = CONFIG_DIR / ".env"
    dst_env = root / ".env"

    if dst_env.exists():
        content = dst_env.read_text(encoding="utf-8")
        if credentials is None:
            # Sharing disabled — set localhost-only, clear auth
            content = _upsert_env_value(content, "SERVER_NAME", "127.0.0.1")
            content = _upsert_env_value(content, "ACESTEP_AUTH_USERNAME", "")
            content = _upsert_env_value(content, "ACESTEP_AUTH_PASSWORD", "")
            content = content.replace("\r\n", "\n").replace("\n", "\r\n")
            dst_env.write_bytes(content.encode("utf-8"))
            ok(".env updated: remote sharing disabled (localhost only, no auth)")
        elif credentials:
            # Credentials provided — update auth + ensure remote access
            username, password = credentials
            content = _upsert_env_value(content, "SERVER_NAME", "0.0.0.0")
            content = _upsert_env_value(content, "ACESTEP_AUTH_USERNAME", username)
            content = _upsert_env_value(content, "ACESTEP_AUTH_PASSWORD", password)
            content = content.replace("\r\n", "\n").replace("\n", "\r\n")
            dst_env.write_bytes(content.encode("utf-8"))
            ok(f".env updated with new credentials (other settings preserved)")

        # Update LM backend setting in existing .env if needed
        if force_pt_backend:
            content = dst_env.read_text(encoding="utf-8")
            if re.search(r'^ACESTEP_LM_BACKEND\s*=', content, re.MULTILINE):
                content = re.sub(
                    r'^ACESTEP_LM_BACKEND\s*=.*$',
                    'ACESTEP_LM_BACKEND=pt',
                    content, flags=re.MULTILINE,
                )
            else:
                content = content.rstrip() + '\n\n# PyTorch backend (no CUDA graphs available — vLLM would be slower)\nACESTEP_LM_BACKEND=pt\n'
            content = content.replace("\r\n", "\n").replace("\n", "\r\n")
            dst_env.write_bytes(content.encode("utf-8"))
            ok(".env updated: ACESTEP_LM_BACKEND=pt")

    elif src_env.exists():
        content = src_env.read_text(encoding="utf-8")
        # Handle sharing/credentials
        if credentials is None:
            # Sharing disabled — localhost only, clear auth
            content = _upsert_env_value(content, "SERVER_NAME", "127.0.0.1")
            content = _upsert_env_value(content, "ACESTEP_AUTH_USERNAME", "")
            content = _upsert_env_value(content, "ACESTEP_AUTH_PASSWORD", "")
        elif credentials:
            username, password = credentials
            content = _upsert_env_value(content, "SERVER_NAME", "0.0.0.0")
            content = _upsert_env_value(content, "ACESTEP_AUTH_USERNAME", username)
            content = _upsert_env_value(content, "ACESTEP_AUTH_PASSWORD", password)
        # Inject LM backend if needed
        if force_pt_backend:
            content += '\n# PyTorch backend (no CUDA graphs available — vLLM would be slower)\nACESTEP_LM_BACKEND=pt\n'
        # Ensure Windows line endings
        content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        dst_env.write_bytes(content.encode("utf-8"))
        ok(f"Installed: .env -> {dst_env}")
    else:
        err(f"Source not found: {src_env}")

    # Apply profile settings to .env
    if profile and dst_env.exists():
        content = dst_env.read_text(encoding="utf-8")
        content = _upsert_env_value(content, "ACESTEP_CONFIG_PATH", profile["dit_model"])
        content = _upsert_env_value(content, "ACESTEP_LM_MODEL_PATH", profile["lm_model"])
        content = _upsert_env_value(content, "ACESTEP_BATCH_SIZE", str(profile["batch_size"]))
        # Add profile comment
        if "# Profile:" not in content:
            content = content.rstrip() + f'\n\n# Profile: {profile["name"]} (auto-configured by installer v11)\n'
        else:
            content = re.sub(
                r'^# Profile:.*$',
                f'# Profile: {profile["name"]} (auto-configured by installer v11)',
                content, flags=re.MULTILINE,
            )
        content = content.replace("\r\n", "\n").replace("\n", "\r\n")
        dst_env.write_bytes(content.encode("utf-8"))
        ok(f".env configured for {profile['name']} profile")
        info(f"  DiT model:  {profile['dit_model']}")
        info(f"  LM model:   {profile['lm_model']}")
        info(f"  Batch size:  {profile['batch_size']}")

    # Start_Custom.bat
    src_bat = CONFIG_DIR / "Start_Custom.bat"
    dst_bat = root / "Start_Custom.bat"
    if src_bat.exists():
        backup_if_exists(dst_bat)
        shutil.copy2(src_bat, dst_bat)
        ok(f"Installed: Start_Custom.bat -> {dst_bat}")
    else:
        err(f"Source not found: {src_bat}")

    # update.bat → install root
    src_update = CONFIG_DIR / "update.bat"
    dst_update = root / "update.bat"
    if src_update.exists():
        shutil.copy2(src_update, dst_update)
        ok(f"Installed: update.bat -> {dst_update}")

    # patch_only.bat → install root
    src_patchonly = CONFIG_DIR / "patch_only_root.bat"
    dst_patchonly = root / "patch_only.bat"
    if src_patchonly.exists():
        shutil.copy2(src_patchonly, dst_patchonly)
        ok(f"Installed: patch_only.bat -> {dst_patchonly}")

    # Copy patch toolkit to _custom_patches/ inside install root
    # so patch_only.bat and update.bat can work standalone
    patch_dest = root / "_custom_patches"
    patch_dest.mkdir(parents=True, exist_ok=True)

    # install.py (the patcher itself)
    src_installer = SCRIPT_DIR / "install.py"
    if src_installer.exists():
        shutil.copy2(src_installer, patch_dest / "install.py")

    # patches/ directory
    if PATCHES_DIR.exists():
        dst_patches = patch_dest / "patches"
        if dst_patches.exists():
            shutil.rmtree(dst_patches)
        shutil.copytree(PATCHES_DIR, dst_patches)

    # config/ directory (for Start_Custom.bat template)
    if CONFIG_DIR.exists():
        dst_config = patch_dest / "config"
        if dst_config.exists():
            shutil.rmtree(dst_config)
        shutil.copytree(CONFIG_DIR, dst_config)

    ok(f"Installed: patch toolkit -> {patch_dest}")


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — File-replacement patches (VRAM fixes + batch_navigation)
# ═══════════════════════════════════════════════════════════════════════════

def apply_file_patches(root: Path) -> dict:
    header("Step 5 -- Applying file-replacement patches (VRAM + navigation)")
    results = {}
    patch_map = get_patch_map(root)

    for patch_name, dst_path in patch_map.items():
        src_path = PATCHES_DIR / patch_name
        sentinel = PATCH_SENTINELS[patch_name]

        if not src_path.exists():
            err(f"{patch_name}: patch source not found at {src_path}")
            results[patch_name] = "MISSING_SOURCE"
            continue

        if not dst_path.exists():
            err(f"{patch_name}: destination not found at {dst_path}")
            results[patch_name] = "MISSING_DEST"
            continue

        current = read_file(dst_path)
        if sentinel in current:
            ok(f"{patch_name}: already patched — skipping")
            results[patch_name] = "ALREADY_APPLIED"
            continue

        backup_if_exists(dst_path)

        try:
            shutil.copy2(src_path, dst_path)
            written = read_file(dst_path)
            if sentinel in written:
                ok(f"{patch_name}: patched successfully")
                results[patch_name] = "APPLIED"
            else:
                err(f"{patch_name}: copied but sentinel not found — patch may be incorrect")
                results[patch_name] = "VERIFY_FAILED"
        except Exception as e:
            err(f"{patch_name}: failed — {e}")
            results[patch_name] = "ERROR"

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — GUI display bug fixes (text-level patches)
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Fix A: Disable inner progressive yield in generate_with_progress
# ---------------------------------------------------------------------------
SENTINEL_A = "# GUI_FIX_A: progressive yield disabled"

def fix_A_progressive_yield(root: Path, results: dict):
    label = "Fix_A_progressive_yield"

    path = find_by_content(root, "generate_with_progress")
    if path is None:
        warn(f"{label}: generate_with_progress not found")
        results[label] = "FILE_NOT_FOUND"
        return

    info(f"{label}: {path.relative_to(root)}")
    text = read_file(path)

    if SENTINEL_A in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    lines = text.splitlines(keepends=True)

    func_start = None
    func_indent = 0
    for i, line in enumerate(lines):
        if re.match(r'\s*def generate_with_progress\b', line):
            func_start = i
            func_indent = len(line) - len(line.lstrip())
            break

    if func_start is None:
        warn(f"{label}: def generate_with_progress not found inside {path.name}")
        results[label] = "FUNC_NOT_FOUND"
        return

    modified = False
    in_for_loop = False
    for_loop_indent = 0

    for i in range(func_start + 1, len(lines)):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("def ") and (len(line) - len(stripped)) <= func_indent and i > func_start:
            break

        if re.match(r'\s+for\s+\S+.*\bin\b.*:\s*(?:#.*)?$', line) and not stripped.startswith("#"):
            in_for_loop = True
            for_loop_indent = len(line) - len(stripped)
            continue

        if in_for_loop and stripped and not stripped.startswith("#"):
            if (len(line) - len(stripped)) <= for_loop_indent:
                in_for_loop = False

        if in_for_loop and re.match(r'\s{8,}yield\b', line) and not stripped.startswith("#"):
            indent_str = " " * (len(line) - len(stripped))
            lines.insert(i, indent_str + SENTINEL_A + "\n")
            i += 1
            lines[i] = indent_str + "# " + lines[i].lstrip()
            open_p = lines[i].count("(") - lines[i].count(")")
            j = i + 1
            while open_p > 0 and j < len(lines):
                open_p += lines[j].count("(") - lines[j].count(")")
                ci = " " * (len(lines[j]) - len(lines[j].lstrip()))
                lines[j] = ci + "# " + lines[j].lstrip()
                j += 1
            modified = True
            break

    if not modified:
        warn(f"{label}: inner yield pattern not found — may need manual fix")
        results[label] = "PATTERN_NOT_FOUND"
        return

    backup_if_exists(path)
    write_file(path, "".join(lines))
    ok(f"{label}: patched {path.name}")
    results[label] = "APPLIED"


# ---------------------------------------------------------------------------
# Fix B: Remove Phase 1 / Phase 2 audio overwrite
# ---------------------------------------------------------------------------
SENTINEL_B = "# GUI_FIX_B: phase audio overwrite removed"

def fix_B_phase_audio_overwrite(root: Path, results: dict):
    label = "Fix_B_phase_audio_overwrite"

    path = find_by_content(root, "generate_with_batch_management")
    if path is None:
        warn(f"{label}: generate_with_batch_management not found")
        results[label] = "FILE_NOT_FOUND"
        return

    info(f"{label}: {path.relative_to(root)}")
    text = read_file(path)

    if SENTINEL_B in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    lines = text.splitlines(keepends=True)
    modified = False

    for i, line in enumerate(lines):
        if re.search(r'#\s*[Pp]hase\s*1\b', line):
            phase_indent = len(line) - len(line.lstrip())
            block_start = i
            block_end = i
            j = i + 1
            while j < len(lines):
                jl = lines[j]
                js = jl.lstrip()
                ji = len(jl) - len(js)
                if re.search(r'#\s*[Pp]hase\s*2\b', jl):
                    k = j + 1
                    while k < len(lines):
                        kl = lines[k]
                        ks = kl.lstrip()
                        ki = len(kl) - len(ks)
                        if ks and ki <= phase_indent and not ks.startswith("#"):
                            break
                        k += 1
                    block_end = k - 1
                    break
                if js and ji <= phase_indent and not js.startswith("#"):
                    block_end = j - 1
                    break
                block_end = j
                j += 1

            for k in range(block_start, block_end + 1):
                orig = lines[k]
                oi = " " * (len(orig) - len(orig.lstrip()))
                if not orig.lstrip().startswith("#"):
                    lines[k] = oi + "# " + orig.lstrip()

            lines.insert(block_start, " " * phase_indent + SENTINEL_B + "\n")
            modified = True
            break

    if not modified:
        warn(f"{label}: Phase 1/2 block not found — may need manual fix")
        results[label] = "PATTERN_NOT_FOUND"
        return

    backup_if_exists(path)
    write_file(path, "".join(lines))
    ok(f"{label}: patched {path.name}")
    results[label] = "APPLIED"


# ---------------------------------------------------------------------------
# Fix C: Normalise output_dir path separators for Gradio 6 on Windows
# ---------------------------------------------------------------------------
SENTINEL_C = "# GUI_FIX_C: path normalised for Gradio 6"

def fix_C_path_separators(root: Path, results: dict):
    label = "Fix_C_path_separators"

    candidates = [
        root / "acestep" / "acestep_v15_pipeline.py",
        root / "acestep_v15_pipeline.py",
    ]
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        path = find_by_content(root, "demo.launch(")

    if path is None:
        warn(f"{label}: pipeline file not found")
        results[label] = "FILE_NOT_FOUND"
        return

    info(f"{label}: {path.relative_to(root)}")
    text = read_file(path)

    if SENTINEL_C in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    lines = text.splitlines(keepends=True)
    modified = False

    for i, line in enumerate(lines):
        # Match "output_dir = ..." but ONLY string / os.path / Path expressions,
        # NOT Gradio components (gr.Textbox etc.) — calling .replace() on a
        # gr.Textbox would raise AttributeError and break the UI.
        if not re.match(r'\s*output_dir\s*=', line):
            continue
        rhs = line.split("=", 1)[1].strip() if "=" in line else ""
        if '.replace(' in line:
            continue
        if re.match(r'gr\.', rhs) or 'Textbox' in rhs or 'gr.' in rhs:
            continue
        if not rhs or rhs.startswith("#"):
            continue

        indent = " " * (len(line) - len(line.lstrip()))
        lines.insert(i + 1,
            indent + 'output_dir = str(output_dir).replace("\\\\", "/")  ' + SENTINEL_C + "\n"
        )
        modified = True
        break

    if not modified:
        warn(f"{label}: output_dir string assignment not found — skipping path fix")

    # Add allowed_paths to demo.launch() if not already present
    text2 = "".join(lines) if modified else text
    launch_patched = False
    if "demo.launch(" in text2 and "allowed_paths" not in text2:
        safe_allowed = (
            "os.environ.get('ACESTEP_OUTPUT_DIR', str(Path(__file__).parent.parent / 'outputs'))"
        )
        text2 = text2.replace(
            "demo.launch(",
            f"demo.launch(allowed_paths=[{safe_allowed}],  # GUI_FIX_C\n        ",
            1,
        )
        if "import os" not in text2:
            text2 = "import os\n" + text2
        if "from pathlib import Path" not in text2 and "import pathlib" not in text2:
            text2 = "from pathlib import Path\n" + text2
        info(f"{label}: allowed_paths added to demo.launch()")
        launch_patched = True

    if modified or launch_patched:
        backup_if_exists(path)
        write_file(path, text2)
        ok(f"{label}: patched {path.name}")
        results[label] = "APPLIED"
    else:
        results[label] = "PATTERN_NOT_FOUND"


# ---------------------------------------------------------------------------
# Fix D: Batch visibility — samples 3 & 4 visible on first load
# ---------------------------------------------------------------------------
SENTINEL_D1 = "# VISIBILITY_FIX: default_batch_size parameter added"
SENTINEL_D2 = "# VISIBILITY_FIX: initial column visibility respects default_batch_size"
SENTINEL_D3 = "# VISIBILITY_FIX: pipeline passes default_batch_size to create_results_section"

def fix_D1_result_py(root: Path, results: dict):
    """Patch result.py: add default_batch_size param and use it for initial visibility."""
    label = "Fix_D1_result_py"

    result_path = root / "acestep" / "ui" / "gradio" / "result.py"
    if not result_path.exists():
        # Search for it
        result_path = find_by_content(root, "create_results_section")
        if result_path is None:
            err(f"{label}: result.py not found")
            results[label] = "FILE_NOT_FOUND"
            return

    info(f"{label}: {result_path.relative_to(root)}")
    text = read_file(result_path)

    # ── Repair check: detect broken inline sentinel from v3 ──────────
    # The old patch put "# VISIBILITY_FIX:..." inline after the expression,
    # which commented out closing parentheses and caused a SyntaxError.
    # Pattern: visible=(i <= default_batch_size)  # VISIBILITY_FIX:...))
    broken_inline = re.search(
        r'visible=\(i <= default_batch_size\)\s*#\s*VISIBILITY_FIX:[^\n]*',
        text,
    )
    if broken_inline:
        warn(f"{label}: detected broken inline sentinel from prior patch — repairing")
        # Remove the inline comment, keep just the expression
        text = re.sub(
            r'(visible=\(i <= default_batch_size\))\s*#\s*VISIBILITY_FIX:[^\n]*',
            r'\1',
            text,
        )
        # Remove any old standalone sentinel lines too
        text = re.sub(r'[ \t]*# VISIBILITY_FIX: initial column visibility[^\n]*\n', '', text)
        # Now re-insert sentinel properly on its own line
        lines = text.splitlines(keepends=True)
        for idx, line in enumerate(lines):
            if "visible=(i <= default_batch_size)" in line:
                indent = " " * (len(line) - len(line.lstrip()))
                lines.insert(idx, indent + SENTINEL_D2 + "\n")
                break
        text = "".join(lines)
        backup_if_exists(result_path)
        write_file(result_path, text)
        ok(f"{label}: repaired broken sentinel — syntax error fixed")
        results[label] = "APPLIED"
        return

    if SENTINEL_D1 in text and SENTINEL_D2 in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    backup_if_exists(result_path)

    # a) Add default_batch_size param to function signature
    OLD_SIG = "def create_results_section(dit_handler) -> dict:"
    NEW_SIG = (
        "def create_results_section(dit_handler, default_batch_size=2) -> dict:"
        f"  {SENTINEL_D1}"
    )

    if OLD_SIG not in text:
        if "def create_results_section(dit_handler, default_batch_size" in text:
            info(f"{label}: signature already has default_batch_size")
        else:
            err(f"{label}: function signature not found — manual fix required")
            results[label] = "SIGNATURE_NOT_FOUND"
            return
    else:
        text = text.replace(OLD_SIG, NEW_SIG, 1)
        ok(f"{label}: function signature updated")

    # b) Fix the visibility expression
    #    IMPORTANT: sentinel must NOT go inline — it would comment out closing parens.
    #    Instead we insert the sentinel as a standalone comment line above.
    OLD_VIS = "visible=(i <= 2)"
    NEW_VIS = "visible=(i <= default_batch_size)"

    if OLD_VIS not in text:
        if "visible=(i <= default_batch_size)" in text:
            info(f"{label}: visibility expression already updated")
        else:
            err(f"{label}: 'visible=(i <= 2)' not found — manual fix required")
            results[label] = "VISIBILITY_EXPR_NOT_FOUND"
            return
    else:
        lines = text.splitlines(keepends=True)
        in_func = False
        patched = False
        for idx, line in enumerate(lines):
            if "def create_results_section" in line:
                in_func = True
            if in_func and OLD_VIS in line and not patched:
                indent = " " * (len(line) - len(line.lstrip()))
                lines[idx] = line.replace(OLD_VIS, NEW_VIS, 1)
                # Insert sentinel as its own line above, not inline
                lines.insert(idx, indent + SENTINEL_D2 + "\n")
                patched = True
        if patched:
            text = "".join(lines)
        else:
            err(f"{label}: could not locate 'visible=(i <= 2)' inside create_results_section")
            results[label] = "VISIBILITY_PATCH_FAILED"
            return
        ok(f"{label}: visibility expression updated")

    write_file(result_path, text)
    results[label] = "APPLIED"


def fix_D2_pipeline_caller(root: Path, results: dict):
    """Patch the pipeline file: pass default_batch_size to create_results_section()."""
    label = "Fix_D2_pipeline_caller"

    pipe_file = None
    for f in (root / "acestep").rglob("*.py"):
        try:
            text = read_file(f)
            if "create_results_section" in text and "demo" in text and f.name != "result.py":
                pipe_file = f
                break
        except Exception:
            pass

    if pipe_file is None:
        warn(f"{label}: pipeline file not found")
        results[label] = "FILE_NOT_FOUND"
        return

    info(f"{label}: {pipe_file.relative_to(root)}")
    text = read_file(pipe_file)

    if SENTINEL_D3 in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    # Check if already passing the param (without our sentinel)
    if "create_results_section(dit_handler, default_batch_size" in text:
        info(f"{label}: already passes default_batch_size (sentinel missing)")
        ok(f"{label}: functionally correct — no change needed")
        results[label] = "ALREADY_CORRECT"
        return

    pattern = re.compile(
        r'([ \t]*)'
        r'(\w+\s*=\s*)?'
        r'create_results_section\(\s*dit_handler\s*\)',
        re.MULTILINE,
    )

    match = pattern.search(text)
    if not match:
        warn(f"{label}: 'create_results_section(dit_handler)' call not found")
        results[label] = "PATTERN_NOT_FOUND"
        return

    backup_if_exists(pipe_file)

    indent     = match.group(1)
    assignment = match.group(2) or ""
    replacement = (
        f"{indent}_default_bs = (init_params or {{}}).get('default_batch_size', 2)"
        f"  {SENTINEL_D3}\n"
        f"{indent}{assignment}create_results_section(dit_handler, default_batch_size=_default_bs)"
    )

    text = text.replace(match.group(0), replacement, 1)
    write_file(pipe_file, text)

    if "default_batch_size=_default_bs" in read_file(pipe_file):
        ok(f"{label}: patched {pipe_file.name}")
        results[label] = "APPLIED"
    else:
        err(f"{label}: patch written but verification failed")
        results[label] = "VERIFY_FAILED"


# ---------------------------------------------------------------------------
# Fix E: Key / Time Signature / Vocal Language visibility (REWRITTEN in v4, sentinel fix in v5)
#
# v3 Bug:  Injected a nuclear CSS rule targeting ALL .gradio-container .form
#          elements, which broke visibility management for the entire UI.
#          Also matched on bare variable name 'key' which is far too generic.
#
# v4/v5 Fix:  Only patches explicit visible=False on component definitions whose
#          label= argument contains "Key", "Time Signature", or "Vocal Language".
#          No CSS injection.  No event commenting.
# ---------------------------------------------------------------------------
SENTINEL_E = "# GUI_FIX_E_V4: optional param visibility fixed"

# We match on the label= string inside Gradio component constructors, not on
# the Python variable name.  This avoids false positives on variables like
# 'key' that are extremely common in Python.
TARGET_LABELS = [
    "key",
    "time_signature",
    "time signature",
    "time sig",
    "vocal_language",
    "vocal language",
    "vocal lang",
]


def _find_component_end(lines: list, start: int) -> int:
    """Given a line where a component assignment starts, find the closing ')'."""
    open_count = 0
    for i in range(start, len(lines)):
        open_count += lines[i].count("(") - lines[i].count(")")
        if open_count <= 0:
            return i
    return start


def fix_E_optional_params(root: Path, results: dict):
    """Fix visibility of Key / Time Signature / Vocal Language fields."""
    header("Step 7 -- Fix E: Optional Parameters visibility (Key / Time Sig / Vocal Lang)")
    label = "Fix_E_optional_params"

    # Find the UI file(s) that define these components
    # They should contain gr.Dropdown or gr.Textbox and one of our target labels
    candidates = []
    for target in TARGET_LABELS:
        for f in find_all_by_content(root, target):
            text = read_file(f)
            if ("gr.Dropdown" in text or "gr.Textbox" in text or "gr.Radio" in text):
                if f not in candidates:
                    candidates.append(f)

    if not candidates:
        pf = find_by_content(root, "demo.launch(")
        if pf:
            candidates = [pf]

    if not candidates:
        warn(f"{label}: could not locate UI files with optional parameter definitions")
        results[label] = "FILE_NOT_FOUND"
        return

    any_modified = False
    for path in candidates:
        file_label = f"Fix_E ({path.name})"
        text = read_file(path)

        if SENTINEL_E in text:
            ok(f"{file_label}: already patched — skipping")
            results[f"Fix_E_{path.name}"] = "ALREADY_APPLIED"
            continue

        lines = text.splitlines(keepends=True)
        modified = False
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for lines that are Gradio component assignments
            # Pattern: varname = gr.Dropdown( or gr.Textbox( or gr.Radio(
            comp_match = re.match(
                r'^(?P<indent>[ \t]*)\w+\s*=\s*(?:gr\.Dropdown|gr\.Textbox|gr\.Radio)\s*\(',
                line,
            )
            if not comp_match:
                i += 1
                continue

            # Found a component — get the full constructor call
            comp_end = _find_component_end(lines, i)
            block = "".join(lines[i : comp_end + 1])

            # Check if this component's label= matches one of our targets
            label_match = re.search(r'label\s*=\s*["\']([^"\']*)["\']', block)
            if label_match is None:
                # Also check for label via t() / i18n function
                label_match = re.search(r'label\s*=\s*t\(\s*["\']([^"\']*)["\']', block)

            if label_match is None:
                i = comp_end + 1
                continue

            label_value = label_match.group(1).lower()

            # Check if this label matches any of our targets
            is_target = False
            for target in TARGET_LABELS:
                if target in label_value:
                    is_target = True
                    break

            if not is_target:
                i = comp_end + 1
                continue

            # This IS one of our target components.  Check its visibility.
            if "visible=True" in block or "visible = True" in block:
                # Already correct
                i = comp_end + 1
                continue

            if "visible=False" in block or "visible = False" in block:
                # Change explicit False to True
                new_block = re.sub(r'visible\s*=\s*False', "visible=True", block)
                lines[i : comp_end + 1] = new_block.splitlines(keepends=True)
                ok(f"  {label_value}: visible=False -> visible=True")
                modified = True
                i += 1
                continue

            # Check for conditional visible= (any expression that isn't True/False)
            vis_match = re.search(r'visible\s*=\s*([^,)\n]+)', block)
            if vis_match:
                expr = vis_match.group(1).strip()
                if expr not in ("True", "False", "true", "false"):
                    warn(f"  {label_value}: conditional visible=({expr}) — forcing True")
                    new_block = re.sub(r'visible\s*=\s*[^,)\n]+', "visible=True", block)
                    lines[i : comp_end + 1] = new_block.splitlines(keepends=True)
                    modified = True
                    i += 1
                    continue

            # No visible= argument at all — it defaults to True in Gradio,
            # so no action needed.  (v3 bug: it added visible=True here
            # unnecessarily, risking malformed code.)
            i = comp_end + 1

        if modified:
            # Add sentinel at the top of the file
            sentinel_line = f"# {SENTINEL_E}\n"
            lines.insert(0, sentinel_line)
            backup_if_exists(path)
            write_file(path, "".join(lines))
            ok(f"{file_label}: patched")
            results[f"Fix_E_{path.name}"] = "APPLIED"
            any_modified = True
        else:
            ok(f"{file_label}: no visibility issues found — no changes needed")
            results[f"Fix_E_{path.name}"] = "NO_CHANGE_NEEDED"

    if not any_modified:
        info("Fix E: All component definitions look correct.")
        info("  If fields still appear broken, try: clear browser cache, zoom 100%.")


# ---------------------------------------------------------------------------
# Fix E Cleanup: Remove broken CSS from previous v3 patches
# ---------------------------------------------------------------------------
def fix_E_cleanup_v3_css(root: Path, results: dict):
    """Remove the nuclear CSS injection from v3 if present."""
    label = "Fix_E_cleanup_v3_css"

    # Find files with the old v3 CSS sentinel
    old_sentinel = "# GUI_FIX_E: optional params visibility fixed"
    affected = find_all_by_content(root, old_sentinel)

    if not affected:
        # Also check for the CSS constant name
        affected = find_all_by_content(root, "_FIX_E_CSS")

    if not affected:
        ok(f"{label}: no v3 CSS artifacts found — clean install")
        results[label] = "NO_CHANGE_NEEDED"
        return

    for path in affected:
        info(f"{label}: cleaning v3 CSS from {path.name}")
        text = read_file(path)
        original = text

        # Remove the CSS constant definition
        # Pattern: _FIX_E_CSS = """..."""  # GUI_FIX_E: ...
        text = re.sub(
            r'_FIX_E_CSS\s*=\s*"""[\s\S]*?"""\s*#[^\n]*\n?',
            '',
            text,
        )

        # Remove css=_FIX_E_CSS from gr.Blocks() calls
        text = re.sub(r'css\s*=\s*_FIX_E_CSS\s*,\s*', '', text)
        text = re.sub(r'css\s*=\s*_FIX_E_CSS\s*\+\s*', 'css=', text)
        text = re.sub(r'\+\s*_FIX_E_CSS', '', text)

        # Remove old sentinel comments
        text = re.sub(r'#\s*GUI_FIX_E: optional params visibility fixed\s*\n?', '', text)

        # Remove GUI_FIX_E2 commented-out lines — uncomment them to restore originals
        lines = text.splitlines(keepends=True)
        restored = []
        for line in lines:
            if "# GUI_FIX_E2 (commented): " in line:
                # Restore the original line
                indent = " " * (len(line) - len(line.lstrip()))
                original_content = line.split("# GUI_FIX_E2 (commented): ", 1)[1]
                restored.append(indent + original_content)
            else:
                restored.append(line)
        text = "".join(restored)

        if text != original:
            backup_if_exists(path)
            write_file(path, text)
            ok(f"{label}: cleaned v3 CSS artifacts from {path.name}")
            results[f"{label}_{path.name}"] = "CLEANED"
        else:
            ok(f"{label}: {path.name} — no changes needed")
            results[f"{label}_{path.name}"] = "NO_CHANGE_NEEDED"


# ---------------------------------------------------------------------------
# Fix F: DiT inference steps default (32 → 50 for non-turbo models)
# ---------------------------------------------------------------------------
SENTINEL_F = "# GUI_FIX_F: inference_steps default set to 50"

def fix_F_inference_steps_default(root: Path, results: dict):
    """Change the default DiT inference steps from 32 to 50 for non-turbo (base/SFT) models."""
    label = "Fix_F_inference_steps_default"

    target_file = root / "acestep" / "ui" / "gradio" / "events" / "generation" / "model_config.py"
    if not target_file.exists():
        target_file = find_by_content(root, "inference_steps_value")
        if target_file is None:
            warn(f"{label}: model_config.py not found")
            results[label] = "FILE_NOT_FOUND"
            return

    info(f"{label}: {target_file.relative_to(root)}")
    text = read_file(target_file)

    if SENTINEL_F in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    # Only patch the non-turbo (else) branch value of 32 → 50.
    # The turbo branch uses 8, which must stay unchanged.
    OLD = '"inference_steps_value": 32,'
    NEW = '"inference_steps_value": 50,  ' + SENTINEL_F

    if OLD not in text:
        if '"inference_steps_value": 50' in text:
            ok(f"{label}: already set to 50 — no change needed")
            results[label] = "ALREADY_CORRECT"
        else:
            warn(f"{label}: expected pattern not found — manual fix may be needed")
            results[label] = "PATTERN_NOT_FOUND"
        return

    backup_if_exists(target_file)
    text = text.replace(OLD, NEW, 1)
    write_file(target_file, text)

    if '"inference_steps_value": 50' in read_file(target_file):
        ok(f"{label}: patched model_config.py (32 → 50)")
        results[label] = "APPLIED"
    else:
        err(f"{label}: patch written but verification failed")
        results[label] = "VERIFY_FAILED"


# ---------------------------------------------------------------------------
# Fix G: Configure loguru log level in pipeline entry point
# ---------------------------------------------------------------------------
SENTINEL_G = "# GUI_FIX_G: loguru configured to INFO level"

def fix_G_pipeline_log_level(root: Path, results: dict):
    """Add loguru INFO-level config to acestep_v15_pipeline.py.

    The CLI entry-point (cli.py) calls _configure_logging() which sets loguru
    to INFO, but Start_Custom.bat launches via acestep_v15_pipeline.py which
    has no logging config — so loguru defaults to DEBUG, spamming the console
    with noisy messages on every generation.
    """
    label = "Fix_G_pipeline_log_level"

    target_file = root / "acestep" / "acestep_v15_pipeline.py"
    if not target_file.exists():
        target_file = find_by_content(root, "acestep_v15_pipeline")
        if target_file is None:
            warn(f"{label}: acestep_v15_pipeline.py not found")
            results[label] = "FILE_NOT_FOUND"
            return

    info(f"{label}: {target_file.relative_to(root)}")
    text = read_file(target_file)

    if SENTINEL_G in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    # The logging block goes right after the dotenv loading try/except.
    # Look for the end of that block.
    ANCHOR = "except ImportError:\n    # python-dotenv not installed, skip loading .env\n    pass"

    if ANCHOR not in text:
        # Try a more relaxed match — just the except ImportError block near dotenv
        ANCHOR_ALT = "except ImportError:\n    pass"
        if ANCHOR_ALT not in text:
            warn(f"{label}: could not find dotenv except block — manual fix may be needed")
            results[label] = "PATTERN_NOT_FOUND"
            return
        anchor = ANCHOR_ALT
    else:
        anchor = ANCHOR

    LOG_BLOCK = (
        "\n"
        "\n"
        "# Configure loguru to INFO level so DEBUG messages don't spam the console.\n"
        "# The CLI entry-point (cli.py) does this itself; the pipeline entry-point\n"
        "# used by Start_Custom.bat needs it too.\n"
        f"{SENTINEL_G}\n"
        "try:\n"
        "    from loguru import logger as _loguru_logger\n"
        "    _loguru_logger.remove()\n"
        "    _loguru_logger.add(sys.stderr, level=\"INFO\")\n"
        "except Exception:\n"
        "    pass"
    )

    backup_if_exists(target_file)
    text = text.replace(anchor, anchor + LOG_BLOCK, 1)
    write_file(target_file, text)

    if SENTINEL_G in read_file(target_file):
        ok(f"{label}: patched acestep_v15_pipeline.py (loguru → INFO)")
        results[label] = "APPLIED"
    else:
        err(f"{label}: patch written but verification failed")
        results[label] = "VERIFY_FAILED"


# ---------------------------------------------------------------------------
# Fix H: Tooltip overflow — CSS cap + JS flip for long hover descriptions
# ---------------------------------------------------------------------------
SENTINEL_H = "/* GUI_FIX_H: tooltip overflow fix */"

def fix_H_tooltip_overflow(root: Path, results: dict):
    """Add CSS and JS to make long tooltips scrollable and flip upward near viewport bottom."""
    label = "Fix_H_tooltip_overflow"

    # Find the main interface __init__.py
    target = root / "acestep" / "ui" / "gradio" / "interfaces" / "__init__.py"
    if not target.exists():
        target = find_by_content(root, "create_gradio_interface")
        if target is None:
            warn(f"{label}: interfaces/__init__.py not found")
            results[label] = "FILE_NOT_FOUND"
            return

    info(f"{label}: {target.relative_to(root)}")
    text = read_file(target)

    if SENTINEL_H in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    # Inject JS snippet into head= for tooltip flipping
    JS_SNIPPET = '''
        <script>
        /* Flip tooltips upward when they would overflow the viewport bottom */
        document.addEventListener('mouseover', function(e) {
            var el = e.target.closest('.has-info-container');
            if (!el) return;
            var rect = el.getBoundingClientRect();
            if (rect.bottom > window.innerHeight * 0.65) {
                el.classList.add('tooltip-flip');
            } else {
                el.classList.remove('tooltip-flip');
            }
        });
        </script>
        '''

    CSS_SNIPPET = (
        f"\n        {SENTINEL_H}\n"
        "        /* Cap tooltip height and allow scrolling when content is long */\n"
        "        .has-info-container span[data-testid=\"block-info\"]:hover + div,\n"
        "        .has-info-container span[data-testid=\"block-info\"]:hover + span,\n"
        "        .checkbox-container:hover + div {\n"
        "            max-height: 40vh;\n"
        "            overflow-y: auto;\n"
        "            pointer-events: auto;\n"
        "        }\n"
        "\n"
        "        /* Flip tooltip above when near the bottom of the viewport */\n"
        "        .has-info-container.tooltip-flip span[data-testid=\"block-info\"] + div,\n"
        "        .has-info-container.tooltip-flip span[data-testid=\"block-info\"] + span {\n"
        "            bottom: 100%;\n"
        "            top: auto;\n"
        "            margin-top: 0;\n"
        "            margin-bottom: 6px;\n"
        "        }\n"
    )

    modified = False

    # 1. Add JS to head=
    if "get_audio_player_preferences_head()" in text and "tooltip-flip" not in text:
        text = text.replace(
            "get_audio_player_preferences_head(),",
            'get_audio_player_preferences_head() + """' + JS_SNIPPET + '""",',
            1,
        )
        if 'get_audio_player_preferences_head() + """' not in text:
            # Try without trailing comma
            text = text.replace(
                "get_audio_player_preferences_head()",
                'get_audio_player_preferences_head() + """' + JS_SNIPPET + '"""',
                1,
            )
        modified = True

    # 2. Add CSS block — insert after any existing css= opening
    if "tooltip-flip" not in text or SENTINEL_H not in text:
        # Find the css=""" block inside gr.Blocks and append our rules
        css_marker = 'css="""'
        if css_marker in text:
            # Insert after the first css=""" line
            idx = text.index(css_marker) + len(css_marker)
            text = text[:idx] + CSS_SNIPPET + text[idx:]
            modified = True
        else:
            warn(f"{label}: could not locate css= block — CSS not injected")

    if modified:
        backup_if_exists(target)
        write_file(target, text)
        ok(f"{label}: patched {target.name} (tooltip overflow CSS + JS)")
        results[label] = "APPLIED"
    else:
        warn(f"{label}: no changes made — patterns not matched")
        results[label] = "PATTERN_NOT_FOUND"


# ---------------------------------------------------------------------------
# Fix I: Add info= tooltips to CFG Interval Start/End sliders
# ---------------------------------------------------------------------------
SENTINEL_I = "# GUI_FIX_I: CFG interval info tooltips added"

def fix_I_cfg_interval_tooltips(root: Path, results: dict):
    """Add info= and elem_classes= to CFG Interval Start and End sliders."""
    label = "Fix_I_cfg_interval_tooltips"

    target = root / "acestep" / "ui" / "gradio" / "interfaces" / "generation_advanced_dit_controls.py"
    if not target.exists():
        target = find_by_content(root, "cfg_interval_start")
        if target is None:
            warn(f"{label}: generation_advanced_dit_controls.py not found")
            results[label] = "FILE_NOT_FOUND"
            return

    info(f"{label}: {target.relative_to(root)}")
    text = read_file(target)

    if SENTINEL_I in text:
        ok(f"{label}: already patched — skipping")
        results[label] = "ALREADY_APPLIED"
        return

    modified = False

    # Add info= to cfg_interval_start if missing
    if 'label=t("generation.cfg_interval_start")' in text and 'info=t("generation.cfg_interval_start_info")' not in text:
        text = text.replace(
            'label=t("generation.cfg_interval_start"),',
            'label=t("generation.cfg_interval_start"),\n                info=t("generation.cfg_interval_start_info"),',
            1,
        )
        modified = True

    # Add elem_classes to cfg_interval_start if missing
    if 'label=t("generation.cfg_interval_start")' in text and 'elem_classes=["has-info-container"]' not in text.split('cfg_interval_end')[0]:
        # Find the cfg_interval_start visible= line and add elem_classes after it
        old_start_vis = 'visible=ui_config["cfg_interval_start_visible"],\n            )'
        new_start_vis = 'visible=ui_config["cfg_interval_start_visible"],\n                elem_classes=["has-info-container"],\n            )'
        if old_start_vis in text:
            text = text.replace(old_start_vis, new_start_vis, 1)
            modified = True

    # Add info= to cfg_interval_end if missing
    if 'label=t("generation.cfg_interval_end")' in text and 'info=t("generation.cfg_interval_end_info")' not in text:
        text = text.replace(
            'label=t("generation.cfg_interval_end"),',
            'label=t("generation.cfg_interval_end"),\n                info=t("generation.cfg_interval_end_info"),',
            1,
        )
        modified = True

    # Add elem_classes to cfg_interval_end if missing
    if 'label=t("generation.cfg_interval_end")' in text:
        after_end = text.split('label=t("generation.cfg_interval_end")', 1)
        if len(after_end) > 1 and 'elem_classes=["has-info-container"]' not in after_end[1].split('\n            )')[0]:
            old_end_vis = 'visible=ui_config["cfg_interval_end_visible"],\n            )'
            new_end_vis = 'visible=ui_config["cfg_interval_end_visible"],\n                elem_classes=["has-info-container"],\n            )'
            # Only replace the second occurrence (the one for cfg_interval_end)
            parts = text.split(old_end_vis, 2)
            if len(parts) >= 3:
                text = parts[0] + old_end_vis + parts[1] + new_end_vis + parts[2]
                modified = True
            elif len(parts) == 2:
                text = parts[0] + new_end_vis + parts[1]
                modified = True

    if modified:
        # Add sentinel at top
        text = f"{SENTINEL_I}\n" + text
        backup_if_exists(target)
        write_file(target, text)
        ok(f"{label}: patched {target.name}")
        results[label] = "APPLIED"
    else:
        ok(f"{label}: tooltips already present or patterns not matched")
        results[label] = "NO_CHANGE_NEEDED"


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(all_results: dict):
    header("Summary")

    good   = {"APPLIED", "ALREADY_APPLIED", "ALREADY_CORRECT", "NO_CHANGE_NEEDED", "CLEANED"}
    warn_s = {"PATTERN_NOT_FOUND", "FUNC_NOT_FOUND", "MISSING_DEST",
              "FILE_NOT_FOUND", "NOT_FOUND_CHECK_MANUALLY"}
    fail_s = {"MISSING_SOURCE", "VERIFY_FAILED", "ERROR",
              "SIGNATURE_NOT_FOUND", "VISIBILITY_EXPR_NOT_FOUND", "VISIBILITY_PATCH_FAILED"}

    failures = sum(1 for v in all_results.values() if v in fail_s)
    warnings = sum(1 for v in all_results.values() if v in warn_s)

    print()
    for name, status in all_results.items():
        if status in good:
            ok(f"{name}: {status}")
        elif status in warn_s:
            warn(f"{name}: {status}")
        else:
            err(f"{name}: {status}")

    print()
    if failures == 0 and warnings == 0:
        print(f"  {GREEN}{BOLD}All patches applied successfully.{RESET}")
        print(f"  {GREEN}Launch ACE-Step with: Start_Custom.bat{RESET}")
    elif failures == 0:
        print(f"  {YELLOW}{BOLD}{warnings} fix(es) reported warnings — see above.{RESET}")
        print(f"  {YELLOW}Warnings are safe to ignore if your version lacks those patterns.{RESET}")
        print(f"  {GREEN}Launch ACE-Step with: Start_Custom.bat{RESET}")
    else:
        print(f"  {RED}{BOLD}{failures} patch(es) failed. Review errors above.{RESET}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ACE-Step v1.5 installer + patcher (v11)"
    )
    parser.add_argument(
        "install_dir",
        nargs="?",
        default=None,
        help="Installation directory for ACE-Step",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Only apply patches — skip clone and model download",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["express", "balanced", "quality", "studio"],
        default=None,
        help="Installation profile (skips hardware detection prompt)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(" ACE-Step Custom Setup  (v11 — hardware autodetect)")
    print(f"{'='*60}{RESET}")

    # Determine install directory
    if args.install_dir:
        install_dir = Path(args.install_dir).resolve()
    else:
        # Prompt the user
        default_dir = Path.cwd()
        user_input = input(
            f"\n  Enter ACE-Step install directory\n"
            f"  (press Enter for current directory: {default_dir})\n"
            f"  > "
        ).strip()
        install_dir = Path(user_input).resolve() if user_input else default_dir

    print(f"\n Setup package: {SCRIPT_DIR}")
    print(f" Install dir:   {install_dir}")
    print()

    # ── Verify package integrity ──────────────────────────────────────────
    verify_integrity()

    if args.patch_only:
        print(" Mode: PATCH ONLY (no clone, no model download)")
        print()
        print(" Patches included:")
        print("   — VRAM / scoring fixes (file replacements)")
        print("   — GUI display bug fixes A/B/C")
        print("   — Batch visibility fix D")
        print("   — Optional params visibility fix E (v4+ — rewritten)")
        print("   — v3 CSS cleanup (removes broken CSS from prior patches)")
        print("   — DiT inference steps default fix F (32 → 50)")
        print("   — Pipeline log level fix G (suppress DEBUG spam)")
        print("   — Tooltip overflow fix H (scrollable + flip)")
        print("   — CFG interval tooltip fix I (missing info=)")
        print("   — Improved tooltip descriptions (en.json)")
        print("   — Triton error suppression (llm_inference.py)")
    else:
        print(" Mode: FULL INSTALL (with hardware autodetect)")
        print()
        print(" Steps:")
        print("   0. Detect hardware (GPU, VRAM, system RAM)")
        print("   1. Choose installation profile")
        print("   2. Clone ACE-Step v1.5 from GitHub")
        print("   3. Set up environment (uv sync)")
        print("   4. Download models for chosen profile")
        print("   5. Install .env + Start_Custom.bat")
        print("   6. Apply VRAM + navigation patches")
        print("   7. Apply GUI fixes A/B/C/D/E/F/G/H/I")

    # ── Hardware detection + profile selection (full install only) ─────
    profile = None
    if not args.patch_only:
        if args.profile:
            # CLI-specified profile — skip interactive detection
            profile_map = {p["name"].lower(): p for p in PROFILES}
            profile = profile_map[args.profile]
            ok(f"Using CLI-specified profile: {profile['name']}")
        else:
            # Interactive hardware detection + profile selection
            hardware = detect_hardware()
            profile = prompt_profile_selection(hardware)

    # ── Prompt for credentials ──────────────────────────────────────────
    credentials = prompt_credentials()

    # ── Full install steps ──────────────────────────────────────────────
    if not args.patch_only:
        if not clone_repo(install_dir):
            input("\nPress Enter to close...")
            sys.exit(1)

        python_mode = setup_environment(install_dir)
        if python_mode is None:
            input("\nPress Enter to close...")
            sys.exit(1)

        # Test vLLM compatibility (needs torch from the venv)
        force_pt_backend = test_vllm_compatibility(install_dir, python_mode)

        install_hf_xet(install_dir, python_mode)
        download_models(install_dir, python_mode, profile)
        install_config(install_dir, credentials, force_pt_backend, profile)
    else:
        # Verify the install exists
        if not (install_dir / "acestep").exists():
            err(f"'acestep' folder not found under: {install_dir}")
            err("Run this script from your ACE-Step root, or provide the correct path.")
            input("\nPress Enter to close...")
            sys.exit(1)

        # Determine python mode for the test
        if (install_dir / "python_embeded" / "python.exe").exists():
            python_mode = "embedded"
        else:
            python_mode = "uv"

        # Test vLLM compatibility
        force_pt_backend = test_vllm_compatibility(install_dir, python_mode)

        # Install config files (update credentials + backend, no profile)
        install_config(install_dir, credentials, force_pt_backend)

    # ── Apply patches ─────────────────────────────────────────────────
    all_results = {}

    file_results = apply_file_patches(install_dir)
    all_results.update(file_results)

    header("Step 6 -- GUI display bug fixes (A / B / C / D)")
    fix_A_progressive_yield(install_dir, all_results)
    fix_B_phase_audio_overwrite(install_dir, all_results)
    fix_C_path_separators(install_dir, all_results)
    fix_D1_result_py(install_dir, all_results)
    fix_D2_pipeline_caller(install_dir, all_results)

    # Clean up v3 CSS mess before applying v8 fix
    fix_E_cleanup_v3_css(install_dir, all_results)

    # Apply the rewritten Fix E
    fix_E_optional_params(install_dir, all_results)

    # Fix F: DiT inference steps default (32 → 50)
    fix_F_inference_steps_default(install_dir, all_results)

    # Fix G: Suppress DEBUG log spam in pipeline entry point
    fix_G_pipeline_log_level(install_dir, all_results)

    header("Step 8 -- Tooltip / UX improvements (H / I)")
    # Fix H: Tooltip overflow CSS + JS
    fix_H_tooltip_overflow(install_dir, all_results)

    # Fix I: CFG interval tooltips
    fix_I_cfg_interval_tooltips(install_dir, all_results)

    print_summary(all_results)

    print()
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
