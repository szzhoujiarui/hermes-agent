"""nvidia-smi field isolation for the shared GPU query (issue #120262).

A combined query must not take the name, the memory budget, or backend
selection down with one unsupported or ``N/A`` cell, and a miss must not be
cached in a form that makes the next reader skip a narrower probe that would
have worked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.local_runtime import bootstrap, hardware
from hermes_cli.local_runtime.binaries import select_backend
from hermes_cli.web_routers import local_models
from hermes_platform.host import facts

_WIDE = "memory.total,memory.free,name,pci.device_id,memory.used,utilization.gpu"
_NAME = "name"
_MEMORY = "memory.total,memory.free"


def _fields(argv) -> str:
    return next(arg.split("=", 1)[1] for arg in argv if str(arg).startswith("--query-gpu="))


@pytest.fixture(autouse=True)
def _reset_gpu_query_cache():
    hardware._gpu_query_cache = None
    hardware._smi_path_cache = None
    hardware._pool_probe_cache = None
    yield
    hardware._gpu_query_cache = None
    hardware._smi_path_cache = None
    hardware._pool_probe_cache = None


def _budget_machine(monkeypatch):
    """Discrete-card host: no allocator-pool view, no Vulkan device, modest RAM.

    If the smi reading is dropped, probe_budget classifies this as unified RAM
    (16 GiB, not the card).
    """
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (16 << 30, 8 << 30))
    monkeypatch.setattr(hardware, "_device_pool_view", lambda: None)
    monkeypatch.setattr(hardware, "_accelerator_device", lambda **_kwargs: None)
    monkeypatch.setattr(facts, "gpu_class", lambda: "none")


def test_na_cells_do_not_drop_name_or_memory(monkeypatch):
    """``N/A`` utilization/used/pci must not blank the name or the VRAM budget.

    The same reading is cached for the TTL. The next reader sees those cells,
    not a cached miss, and does not spawn again.
    """
    _budget_machine(monkeypatch)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["creationflags"] == windows_hide_flags()
        assert _fields(argv) == _WIDE
        return SimpleNamespace(
            returncode=0,
            stdout="32768, 28000, Tesla T4, N/A, N/A, N/A\n",
        )

    monkeypatch.setattr(hardware.subprocess, "run", run)

    query = hardware._cached_nvidia_gpu_query()
    assert query is not None
    assert query["gpu_name"] == "Tesla T4"
    assert query["total_bytes"] == 32768 << 20
    assert query["free_bytes"] == 28000 << 20
    assert query["used_bytes"] is None
    assert query["gpu_util_percent"] is None
    assert query["gpu_pci_id"] is None

    budget = hardware.probe_budget(planning=False)
    assert budget.uma is False
    assert budget.total_device_bytes == 32768 << 20
    margin = max(hardware._MARGIN_FLOOR, int((32768 << 20) * hardware._MARGIN_FRACTION))
    assert budget.usable_vram_bytes == max(0, (28000 << 20) - margin)
    assert budget.gpu_name == "Tesla T4"

    facts_out = local_models._nvidia_smi_facts()
    assert facts_out["gpu_name"] == "Tesla T4"
    assert "gpu_util_percent" not in facts_out
    assert "vram_used_bytes" not in facts_out
    assert bootstrap._detect_gpu_vendor() == "nvidia Tesla T4"
    assert select_backend(bootstrap._detect_gpu_vendor(), os_name="win") == "cuda"

    assert hardware._cached_nvidia_gpu_query() == query
    assert len(calls) == 1
    assert hardware._gpu_query_cache[1] is not None


def test_rejected_combined_query_falls_back_to_name_and_memory(monkeypatch):
    """rc≠0 on the wide field set still selects CUDA and budgets the card.

    Name-only and memory-only are what those consumers ran before the merge.
    The merged reading is cached, so a later reader is not stuck on the wide
    query's failure for the TTL.
    """
    _budget_machine(monkeypatch)
    calls = []

    def run(argv, **_kwargs):
        calls.append(_fields(argv))
        if calls[-1] == _WIDE:
            return SimpleNamespace(returncode=1, stdout="")
        if calls[-1] == _NAME:
            # The pre-merge name probe kept the whole line, commas included.
            return SimpleNamespace(returncode=0, stdout="Tesla T4, 16GB\n")
        if calls[-1] == _MEMORY:
            return SimpleNamespace(returncode=0, stdout="32768, 28000\n")
        raise AssertionError(calls[-1])

    monkeypatch.setattr(hardware.subprocess, "run", run)

    vendor = bootstrap._detect_gpu_vendor()
    assert vendor == "nvidia Tesla T4, 16GB"
    assert select_backend(vendor, os_name="win") == "cuda"
    assert calls == [_WIDE, _NAME, _MEMORY]

    budget = hardware.probe_budget(planning=True)
    assert budget.uma is False
    assert budget.total_device_bytes == 32768 << 20
    assert budget.gpu_name == "Tesla T4, 16GB"
    assert calls == [_WIDE, _NAME, _MEMORY]  # cached partial, not a cached miss

    facts_out = local_models._nvidia_smi_facts()
    assert facts_out == {"gpu_name": "Tesla T4, 16GB"}
    assert hardware._gpu_query_cache[1]["gpu_name"] == "Tesla T4, 16GB"


def test_missing_nvidia_smi_reports_no_gpu_not_zeros(monkeypatch):
    """No binary is a cached miss: facts stay empty and the endpoint stays None.

    Reporting 0% / 0 bytes here is what paints "GPU 0%" on a machine with no
    NVIDIA device. Nothing may spawn.
    """
    _budget_machine(monkeypatch)
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: None)

    def run(*_args, **_kwargs):
        raise AssertionError("nvidia-smi must not spawn when the binary is absent")

    monkeypatch.setattr(hardware.subprocess, "run", run)

    assert hardware._cached_nvidia_gpu_query() is None
    assert local_models._nvidia_smi_facts() == {}
    assert hardware._cached_nvidia_gpu_query() is None

    payload = local_models.local_models_hardware()
    assert payload["gpu_name"] is None
    assert payload["gpu_util_percent"] is None
    assert payload["vram_used_bytes"] is None
