"""Live hardware budget probe.

Budget-source rule: discrete cards may trust the device query (measured honest within rounding);
unified-memory devices must budget from OS free physical memory minus headroom — their device
queries have been observed off by 3x in both directions. Every probe here must work under a
stripped PATH — gateway and service sessions don't inherit the interactive environment.
"""

from __future__ import annotations

from contextlib import suppress
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from hermes_cli.local_runtime.devices import GGML_DEVICE_GPU
from hermes_cli.local_runtime.estimator import HardwareBudget
from hermes_platform.host.facts import gpu_class

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# Reserve carved off the card before any grant: the desktop's own co-residents (compositor,
# browser, Electron) measure ~2-2.5 GiB, and a window granted into that space demotes silently
# under WDDM. 7% covers big cards; the 2 GiB floor is what a 512 MiB floor failed to cover (a
# 221K grant measured 31.9/32.6 GiB with the desktop running — 'fits' by the math, demoted in
# reality). Small cards give up window to this; spill mode is their path to big models.
_MARGIN_FLOOR = 2 << 30
_MARGIN_FRACTION = 0.09
# Room left on top of what other programs hold when a model launches: a browser tab or a chat app
# can take another GiB between launch and the next request.
_LAUNCH_HEADROOM = 1 << 30
# UMA headroom: on unified-memory machines the model shares physical memory with the OS and every
# app, so budget from RAM minus this fraction.
_UMA_HEADROOM_FRACTION = 0.20

# Engine-fallback gates for the unified-pool quirk — BOTH must hold, and no discrete card can
# meet either: (1) the allocator's pool exceeds the smi report by well past rounding/ECC slack
# (discrete cards agree within ~2%; carve-out disagreement runs to whole multiples), and (2) the
# pool is system-RAM-sized. The driver's INTEGRATED attribute, when readable, bypasses both gates
# in whichever direction it points.
_POOL_DISAGREEMENT_FACTOR = 1.5
_POOL_RAM_FRACTION = 0.75

# cuDeviceGetAttribute enum: device is integrated with host memory.
_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18

# One probe per process once a device answers (silicon doesn't change); a miss retries after this
# long so a runtime installed mid-session gets picked up by the engine fallback.
_POOL_NEGATIVE_TTL_S = 60.0
_pool_probe_cache: tuple[float, "tuple[int, bool | None] | None"] | None = None

# '  CUDA0: NVIDIA Example Device (1234-core Example GPU) (46464 MiB, 46284 MiB free)'
# — greedy .* pins the LAST parenthesized group, so device names with parentheses parse.
_DEVICE_LINE_RE = re.compile(r"CUDA\d+:.*\((\d+)\s*MiB,\s*\d+\s*MiB free\)\s*$")


def _stdout(*argv: str) -> str:
    return subprocess.run(
        list(argv), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
    ).stdout


_LINUX_RAM_KEYS = frozenset({"MemTotal", "MemAvailable", "MemFree"})


def _linux_meminfo_text() -> str | None:
    """Raw /proc/meminfo, or None when procfs cannot be read."""
    try:
        return Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None


def _linux_ram_from_meminfo(text: str) -> tuple[int, int] | None:
    """(total, available) from meminfo text, or None if it cannot be trusted.

    MemAvailable includes reclaimable page cache, which ``getconf _AVPHYS_PAGES``
    counts as used. Kernels that omit MemAvailable fall back to MemFree. Zero is
    a real available value — a missing-field check must not treat it as absent.
    """
    fields: dict[str, int] = {}
    try:
        for line in text.splitlines():
            name, separator, value = line.partition(":")
            if not separator or name not in _LINUX_RAM_KEYS:
                continue
            parts = value.split()
            if len(parts) != 2 or parts[1] != "kB":
                continue
            fields[name] = int(parts[0]) * 1024
        total = fields.get("MemTotal")
        # Key presence, not truthiness: MemAvailable 0 must not fall through to MemFree.
        if "MemAvailable" in fields:
            available = fields["MemAvailable"]
        else:
            available = fields.get("MemFree")
        if (
            total is None
            or total <= 0
            or available is None
            or not 0 <= available <= total
        ):
            return None
        return total, available
    except (AttributeError, TypeError, ValueError):
        return None


def _ram_bytes() -> tuple[int, int]:
    """(total, available) physical memory, cross-platform stdlib."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = ([("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)]
                        + [(name, ctypes.c_ulonglong) for name in (
                            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                            "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")])

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys, stat.ullAvailPhys
    except (AttributeError, OSError):
        pass
    try:
        if sys.platform == "darwin":
            # macOS getconf has no _PHYS_PAGES/_AVPHYS_PAGES (exit 64) — the POSIX branch would
            # return (0, 0) and every model would read unavailable. sysctl is the platform truth.
            total = int(_stdout("/usr/sbin/sysctl", "-n", "hw.memsize").strip() or 0)
            if total <= 0:
                return 0, 0
            avail = total // 2  # conservative fallback
            with suppress(OSError, ValueError):
                out = _stdout("/usr/bin/vm_stat")
                page_m = re.search(r"page size of (\d+)", out)
                page = int(page_m.group(1)) if page_m else 16384
                # free + inactive + purgeable ≈ reclaimable-on-demand; the speculative pool is
                # dropped by the OS under pressure too.
                pages = sum(int(m.group(1)) for key in (
                    "Pages free", "Pages inactive", "Pages purgeable", "Pages speculative")
                    if (m := re.search(rf"{key}:\s+(\d+)\.", out)))
                if pages > 0:
                    avail = pages * page
            return total, avail
        if sys.platform.startswith("linux"):
            meminfo = _linux_meminfo_text()
            if meminfo is not None:
                linux_ram = _linux_ram_from_meminfo(meminfo)
                if linux_ram is not None:
                    return linux_ram
        # POSIX
        page = int(_stdout("getconf", "PAGE_SIZE") or 4096)
        total = int(_stdout("getconf", "_PHYS_PAGES") or 0) * page
        avail = total // 2  # conservative when _AVPHYS is unavailable
        with suppress(OSError, ValueError):
            avail = int(_stdout("getconf", "_AVPHYS_PAGES") or 0) * page or avail
        return total, avail
    except (OSError, ValueError):
        return 0, 0


# nvidia-smi lives at a fixed path under the driver install; PATH presence varies by session type
# (services and gateways often run minimal environments) and by driver generation (the legacy
# NVSMI dir was never on PATH). Cached: the driver doesn't move mid-process.
_smi_path_cache: "tuple[str | None] | None" = None


def _nvidia_smi_path() -> str | None:
    """Absolute path to nvidia-smi, or None. PATH first (respects user overrides), then the
    driver's known install locations on Windows and WSL."""
    global _smi_path_cache
    if _smi_path_cache is not None:
        return _smi_path_cache[0]
    found = shutil.which("nvidia-smi")
    if found is None and os.name == "nt":
        windir = os.environ.get("SystemRoot", r"C:\Windows")
        candidates = (
            # DCH drivers (every modern install) place it in System32.
            Path(windir) / "System32" / "nvidia-smi.exe",
            # Legacy standalone drivers used NVSMI, never on PATH.
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        )
        found = next((str(c) for c in candidates if c.exists()), None)
    elif found is None and sys.platform.startswith("linux"):
        # The Windows display driver exposes nvidia-smi here inside WSL, but does not guarantee
        # that the directory is present on PATH.
        candidate = Path("/usr/lib/wsl/lib/nvidia-smi")
        if candidate.exists():
            found = str(candidate)
    _smi_path_cache = (found,)
    return found


def _nvidia_vram() -> tuple[int, int, str, int | None] | None:
    """(total bytes, free bytes, name, optional packed PCI ID) from the shared nvidia-smi query, or None.

    Free memory is optional. A driver that reports capacity and ``N/A`` for free must not look
    like "no card" — that budgets a discrete GPU from system RAM. Unknown free uses capacity,
    which is the figure the driver did give.
    """
    query = _cached_nvidia_gpu_query()
    if query is None or query.get("total_bytes") is None:
        return None
    total = query["total_bytes"]
    free = query.get("free_bytes")
    if free is None:
        free = total
    return total, free, query.get("gpu_name") or "", query.get("gpu_pci_id")


_gpu_query_cache: "tuple[float, dict | None] | None" = None
# The statusbar polls /api/local-models/hardware every 5s and the endpoint needs
# name/util/vram; the budget probe needs total/free. One shared query (with a TTL
# shorter than the poll) serves both, so one poll = one nvidia-smi spawn (#120262)
# instead of two, and the spawn carries CREATE_NO_WINDOW so a console-less backend
# never flashes a window (#101895).
_GPU_QUERY_TTL_S = 4.0

# Wide query first. ``pci.device_id`` / ``utilization.gpu`` are the fields older
# drivers reject for the whole process (rc≠0, empty stdout) or answer with N/A.
# Name-only and memory-only are the probes backend selection and VRAM budgeting
# used before the fields were merged; they still have to work on their own.
_WIDE_GPU_QUERY = "memory.total,memory.free,name,pci.device_id,memory.used,utilization.gpu"
_NAME_GPU_QUERY = "name"
_MEMORY_GPU_QUERY = "memory.total,memory.free"

# nvidia-smi prints these instead of a number when a field exists but this driver
# will not answer it. They are not zero.
_SMI_UNAVAILABLE = frozenset({
    "n/a",
    "[n/a]",
    "[not supported]",
    "[insufficient permissions]",
    "[unknown error]",
})


def _smi_int(token: str, *, base: int = 10) -> int | None:
    """Parse one nvidia-smi cell. Unavailable / non-numeric cells are None, not an error."""
    text = token.strip()
    if not text or text.lower() in _SMI_UNAVAILABLE:
        return None
    try:
        return int(text, base)
    except ValueError:
        return None


def _mib_to_bytes(token: str) -> int | None:
    mib = _smi_int(token)
    if mib is None:
        return None
    return mib << 20


def _blank_gpu_query() -> dict:
    return {
        "gpu_name": "",
        "total_bytes": None,
        "free_bytes": None,
        "used_bytes": None,
        "gpu_util_percent": None,
        "gpu_pci_id": None,
    }


def _gpu_query_has_reading(data: dict) -> bool:
    if data.get("gpu_name"):
        return True
    return any(data.get(key) is not None for key in (
        "total_bytes", "free_bytes", "used_bytes", "gpu_util_percent", "gpu_pci_id"))


def _csv_row(stdout: str) -> list[str]:
    return next(csv.reader(stdout.strip().splitlines(), skipinitialspace=True), [])


def _gpu_name_from_smi_stdout(stdout: str) -> str:
    """Name-only nvidia-smi line. The whole line is the name, including commas."""
    line = stdout.strip().splitlines()[0].strip()
    if len(line) >= 2 and line.startswith('"') and line.endswith('"'):
        line = line[1:-1].replace('""', '"')
    if not line or line.lower() in _SMI_UNAVAILABLE:
        return ""
    return line


def _parse_wide_gpu_csv(stdout: str) -> dict:
    """Parse the shared 6-field row. Each cell stands alone — ``N/A`` drops that cell only."""
    row = _csv_row(stdout)
    if len(row) < 6:
        row = row + [""] * (6 - len(row))
    total_mib, free_mib, name, raw_id, used_mib, util = row[:6]
    name = name.strip()
    if name.lower() in _SMI_UNAVAILABLE:
        name = ""
    return {
        "gpu_name": name,
        "total_bytes": _mib_to_bytes(total_mib),
        "free_bytes": _mib_to_bytes(free_mib),
        "used_bytes": _mib_to_bytes(used_mib),
        "gpu_util_percent": _smi_int(util),
        "gpu_pci_id": _smi_int(raw_id, base=16),
    }


def _parse_memory_csv(stdout: str) -> tuple[int | None, int | None]:
    row = _csv_row(stdout)
    if len(row) < 2:
        row = row + [""] * (2 - len(row))
    return _mib_to_bytes(row[0]), _mib_to_bytes(row[1])


def _nvidia_smi_csv(exe: str, fields: str, *, nounits: bool, creationflags: int) -> str | None:
    """stdout of one nvidia-smi query, or None when that field set is rejected."""
    fmt = "csv,noheader,nounits" if nounits else "csv,noheader"
    out = subprocess.run(
        [exe, f"--query-gpu={fields}", f"--format={fmt}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        creationflags=creationflags)
    if out.returncode != 0 or not (out.stdout or "").strip():
        return None
    return out.stdout


def _read_nvidia_gpu_query(exe: str, creationflags: int) -> dict | None:
    """Wide query, then the historical name-only and memory-only probes for whatever it missed.

    A timeout aborts the rest: another 10s probe would stall install/launch. A fast rejection
    (unknown field, rc≠0) still falls through. When the name probe misses too, stop — a
    memory-only probe will not find a device the name probe could not.
    """
    data = _blank_gpu_query()
    try:
        wide = _nvidia_smi_csv(exe, _WIDE_GPU_QUERY, nounits=True, creationflags=creationflags)
        if wide:
            data = _parse_wide_gpu_csv(wide)
        if not data["gpu_name"]:
            name_out = _nvidia_smi_csv(exe, _NAME_GPU_QUERY, nounits=False, creationflags=creationflags)
            if name_out:
                data["gpu_name"] = _gpu_name_from_smi_stdout(name_out)
        if not wide and not data["gpu_name"]:
            return None
        if data["total_bytes"] is None or data["free_bytes"] is None:
            mem = _nvidia_smi_csv(exe, _MEMORY_GPU_QUERY, nounits=True, creationflags=creationflags)
            if mem:
                total, free = _parse_memory_csv(mem)
                if data["total_bytes"] is None:
                    data["total_bytes"] = total
                if data["free_bytes"] is None:
                    data["free_bytes"] = free
    except (OSError, subprocess.TimeoutExpired):
        if not _gpu_query_has_reading(data):
            return None
    if not _gpu_query_has_reading(data):
        return None
    return data


def _cached_nvidia_gpu_query(ttl_s: float = _GPU_QUERY_TTL_S) -> "dict | None":
    """One nvidia-smi read shared by the budget probe, vendor detection, and the hardware endpoint.

    Returns ``dict(gpu_name=, total_bytes=, free_bytes=, used_bytes=, gpu_util_percent=,
    gpu_pci_id=)``. Numeric cells the driver answers with ``N/A`` are None; the other cells
    are still returned. None only when nvidia-smi is absent or the name probe misses too.

    A partial reading is what gets cached. Caching the wide query's failure as None made
    every other reader miss a name-only / memory-only answer for the whole TTL. A true miss
    (no binary, or the name probe failed) is cached so a missing device does not spawn per poll.
    """
    global _gpu_query_cache
    now = time.monotonic()
    if _gpu_query_cache is not None and now - _gpu_query_cache[0] < ttl_s:
        return _gpu_query_cache[1]

    exe = _nvidia_smi_path()
    if exe is None:
        _gpu_query_cache = (now, None)
        return None

    from hermes_cli._subprocess_compat import windows_hide_flags

    data = _read_nvidia_gpu_query(exe, windows_hide_flags())
    _gpu_query_cache = (now, data)
    return data


def _cuda_driver_pool() -> "tuple[int, bool | None] | None":
    """(allocator_total_bytes, integrated_or_None) from the CUDA driver API via ctypes against the
    driver's own DLL/SO — no toolkit, no subprocess, ~ms. INTEGRATED is the vendor's own
    unified-memory declaration; total is the pool the allocator will actually hand out (on
    carve-out devices, several times what nvidia-smi reports)."""
    import ctypes

    for name in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
        try:
            cuda = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        return None
    with suppress(OSError, AttributeError):
        if cuda.cuInit(0) != 0:
            return None
        dev = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return None
        total = ctypes.c_size_t()
        getter = getattr(cuda, "cuDeviceTotalMem_v2", None) or cuda.cuDeviceTotalMem
        if getter(ctypes.byref(total), dev) != 0 or total.value <= 0:
            return None
        integrated: bool | None = None
        attr = ctypes.c_int()
        if cuda.cuDeviceGetAttribute(
                ctypes.byref(attr), _CU_DEVICE_ATTRIBUTE_INTEGRATED, dev) == 0:
            integrated = bool(attr.value)
        return total.value, integrated
    return None


def _configured_engine():
    """The engine the probes below run: the managed server's own when one is up or booting, else
    the installed engine for ``local_runtime.backend``."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.local_runtime.binaries import installed_engine
    from hermes_cli.local_runtime.bootstrap import serving_engine

    engine = serving_engine()
    if engine is not None:
        return engine
    section = load_config_readonly().get("local_runtime") or {}
    return installed_engine(section.get("backend") or "auto")


def _engine_device_pool() -> "tuple[int, bool | None] | None":
    """(engine_total_bytes, None) from the installed runtime's own --list-devices, or None. The
    fallback when the driver API is unreachable: asks the exact binary that will do the
    allocating. Carries no integrated verdict — callers must gate it."""
    with suppress(Exception):  # a probe miss must never block budgeting
        engine = _configured_engine()
        if engine is None:
            return None
        exe = engine.binary
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=30, cwd=str(exe.parent))
        if out.returncode != 0:
            return None
        for line in (out.stdout + out.stderr).splitlines():
            m = _DEVICE_LINE_RE.search(line)
            if m:
                return int(m.group(1)) << 20, None
    return None


def _device_pool_view() -> "tuple[int, bool | None] | None":
    """Best available allocator-side view, cached: a hit is permanent for the process, a miss
    retries after a short TTL (the engine binary can appear mid-session via a pane install)."""
    global _pool_probe_cache
    now = time.monotonic()
    if _pool_probe_cache is not None:
        stamp, view = _pool_probe_cache
        if view is not None or now - stamp < _POOL_NEGATIVE_TTL_S:
            return view
    view = _cuda_driver_pool() or _engine_device_pool()
    _pool_probe_cache = (now, view)
    return view


_accelerator_cache: "dict[str, tuple[float, dict | None]]" = {}


def _accelerator_device(*, fresh: bool = False) -> "dict | None":
    """The device a Vulkan/HIP engine will place layers on, or None (no such engine, probe miss).

    Mirrors llama.cpp's own placement: discrete devices whenever one exists, an integrated one only
    otherwise; among discrete devices the largest. Cached per engine binary like
    ``_device_pool_view`` (a hit lasts the process, a miss retries after the TTL): the hardware
    endpoint polls this every few seconds and each probe initializes the GPU driver. ``fresh`` skips
    the cache read for a launch-time free-memory answer.
    """
    with suppress(Exception):  # a probe miss must never block budgeting
        from hermes_cli.local_runtime.devices import probe_devices

        engine = _configured_engine()
        if engine is None or engine.backend not in ("vulkan", "hip"):
            return None
        key, now = str(engine.binary), time.monotonic()
        cached = _accelerator_cache.get(key)
        if not fresh and cached is not None and (cached[1] is not None or now - cached[0] < _POOL_NEGATIVE_TTL_S):
            return cached[1]
        devices = probe_devices(engine.binary.parent, engine.backend)
        device = max(devices, key=lambda d: (d["type"] == GGML_DEVICE_GPU, d["total"]), default=None)
        _accelerator_cache[key] = (now, device)
        return device
    return None


def _unified_pool_bytes(smi_total: int, ram_total: int) -> int | None:
    """The real pool size when this NVIDIA device is unified memory behind a carve-out, else None.

    The driver's INTEGRATED attribute decides in BOTH directions when readable; only the
    attribute-less engine fallback needs the two numeric gates, both of which must hold.
    """
    view = _device_pool_view()
    if view is None:
        return None
    pool, integrated = view
    if integrated is not None:
        return pool if integrated else None
    if (smi_total > 0 and pool >= int(smi_total * _POOL_DISAGREEMENT_FACTOR)
            and ram_total > 0 and pool >= int(ram_total * _POOL_RAM_FRACTION)):
        return pool
    return None


def _uma_budget(base: int, total: int, *, gpu_name: str = "",
                gpu_pci_id: int | None = None, lazy_reads: bool = True) -> HardwareBudget:
    usable = max(0, int(base * (1 - _UMA_HEADROOM_FRACTION)))
    return HardwareBudget(usable_vram_bytes=usable, total_device_bytes=total,
                          ram_available_bytes=0, uma=True, gpu_name=gpu_name, platform=sys.platform,
                          gpu_pci_id=gpu_pci_id, lazy_reads=lazy_reads)


def probe_budget(*, planning: bool = False) -> HardwareBudget:
    """Construct the budget per the source rules above.

    ``planning=False``: LIVE budget (free VRAM now) for launch-time fit and growth re-grants.
    ``planning=True``: CAPACITY budget (total minus margin) for catalog pricing and quant
    selection — pricing against live-free while a model was loaded made every row read as too
    large. The managed server unloads/relaunches itself, so capacity is real.
    """
    ram_total, ram_avail = _ram_bytes()
    vram = _nvidia_vram()

    # Unified-memory NVIDIA: the CUDA allocator pool is the real capacity. Classification comes
    # from the driver API/engine and must not require nvidia-smi (stripped-PATH sessions lose smi
    # but nvcuda loads via the system loader). Crossing the carve-out costs nothing — it is an OS
    # accounting knob, not a GPU limit. Deliberately NOT clamped to OS RAM: carved-out memory is
    # invisible to GlobalMemoryStatusEx, so a RAM clamp would throw away exactly that capacity.
    unified = _unified_pool_bytes(vram[0] if vram else 0, ram_total)
    if unified is not None:
        logger.info(
            "unified-memory NVIDIA device: allocator pool %.1f GiB "
            "(nvidia-smi carve-out: %s); budgeting from the pool",
            unified / _GIB,
            f"{vram[0] / _GIB:.1f} GiB" if vram else "unavailable")
        if planning:
            base = unified
        else:
            # Live: dedicated-free plus what the OS can still give. smi's free saturates at the
            # carve-out so this under-counts a bit — the safe direction (the pool edge is a
            # measured soft cliff: decode collapses ~3.5x when concurrent demand hits it).
            live = (vram[1] + ram_avail) if vram else ram_avail
            base = min(unified, live)
        return _uma_budget(base, unified, gpu_name=vram[2] if vram else "",
                           gpu_pci_id=vram[3] if vram else None)

    if vram is None:
        device = _accelerator_device()
        if device is not None and device["type"] == GGML_DEVICE_GPU:
            # Discrete AMD/Intel card behind a Vulkan/HIP engine: its own memory, with system RAM
            # as spill. Capacity serves the polled live view; launch_budget reads free memory.
            total = device["total"]
            margin = max(_MARGIN_FLOOR, int(total * _MARGIN_FRACTION))
            return HardwareBudget(usable_vram_bytes=max(0, total - margin), total_device_bytes=total,
                                  ram_available_bytes=ram_total if planning else ram_avail,
                                  uma=False, gpu_name=device["description"], platform=sys.platform)
        # Integrated GPU, Metal, CPU, or no answer: budget from RAM as unified memory. llama.cpp
        # loads lazy tensors up front on an AMD/Intel integrated GPU, where reading them on demand
        # halved prefill (#28160).
        return _uma_budget(ram_total if planning else ram_avail, ram_total,
                           lazy_reads=gpu_class() not in ("amd", "intel"))

    total, free, gpu_name, gpu_pci_id = vram
    margin = max(_MARGIN_FLOOR, int(total * _MARGIN_FRACTION))
    return HardwareBudget(usable_vram_bytes=max(0, (total if planning else free) - margin),
                          total_device_bytes=total,
                          ram_available_bytes=ram_total if planning else ram_avail,
                          uma=False, gpu_name=gpu_name, platform=sys.platform, gpu_pci_id=gpu_pci_id)


def launch_budget(capacity: HardwareBudget, *, own_bytes: int = 0) -> HardwareBudget | None:
    """Capacity less what other programs hold on the card right now, or None when that can't apply.

    Capacity assumes other programs hold no more than the fixed margin. Beside a heavier desktop the
    capacity-sized window doesn't fit, and Windows pages part of the model to host memory without
    an error: a 32 GiB card with 4.5-6.3 GiB held by other apps decoded at ~24 tok/s at the
    capacity window, and at ~90 tok/s at a window sized from free memory.

    ``own_bytes`` is what the managed server holds now and frees before the new instance loads.
    A stopped server's memory reads as free by the time its process has exited (measured on
    Windows: free memory was fully back at the first reading after exit). None for unified memory
    (its live budget is already free memory) and when the device query fails, so callers keep the
    capacity plan.
    """
    if capacity.uma:
        return None
    vram = _nvidia_vram()
    if vram is not None:
        total, free = vram[0], vram[1]
    else:
        # A discrete card behind Vulkan/HIP: the driver's free count includes other programs.
        device = _accelerator_device(fresh=True)
        if device is None or device["type"] != GGML_DEVICE_GPU:
            return None
        total, free = device["total"], device["free"]
    others = max(0, total - free - max(0, own_bytes))
    usable = min(capacity.usable_vram_bytes, max(0, total - others - _LAUNCH_HEADROOM))
    return replace(capacity, usable_vram_bytes=usable)
