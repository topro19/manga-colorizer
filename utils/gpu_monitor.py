"""
GPU identification and telemetry helpers for Windows / AMD Radeon.
"""

from __future__ import annotations

import subprocess
import json
from typing import Dict, Optional, Tuple


def get_windows_gpu_info() -> Dict[str, str]:
    """
    Query Windows CIM / WMI for installed video controller information.
    Prioritizes dedicated AMD Radeon controllers.
    """
    info = {
        "name": "Unknown GPU",
        "vram_gb": "Unknown",
        "driver_version": "Unknown",
        "is_amd": False,
    }

    try:
        cmd = "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json"
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                data = [data]

            # Find AMD GPU or the adapter with the largest VRAM
            best_gpu = None
            max_ram = 0

            for entry in data:
                name = entry.get("Name", "")
                ram = entry.get("AdapterRAM") or 0
                if "Virtual" in name or "Microsoft Basic" in name:
                    continue
                if "Radeon" in name or "AMD" in name:
                    best_gpu = entry
                    break
                if ram > max_ram:
                    max_ram = ram
                    best_gpu = entry

            if not best_gpu and data:
                best_gpu = data[0]

            if best_gpu:
                info["name"] = best_gpu.get("Name", "Unknown GPU")
                ram_bytes = best_gpu.get("AdapterRAM") or 0
                if ram_bytes > 0:
                    info["vram_gb"] = f"{ram_bytes / (1024**3):.1f} GB"
                    # For RX 9060 XT 16GB, AdapterRAM might report 4GB due to 32-bit UINT limits in legacy WMI
                    if "9060 XT" in info["name"] or "RX 90" in info["name"]:
                        info["vram_gb"] = "16.0 GB"
                info["driver_version"] = best_gpu.get("DriverVersion", "Unknown")
                info["is_amd"] = "AMD" in info["name"] or "Radeon" in info["name"]
    except Exception:
        pass

    return info


def get_active_provider_info(providers: list[str]) -> Tuple[str, str, bool]:
    """
    Format human-readable provider status string and boolean indicating GPU acceleration.
    Returns: (badge_text, detailed_label, is_gpu_active)
    """
    gpu_info = get_windows_gpu_info()
    gpu_name = gpu_info["name"]

    if "DmlExecutionProvider" in providers:
        badge = "DirectML (GPU)"
        detailed = f"GPU: {gpu_name} / DirectML"
        return badge, detailed, True
    elif "ROCMExecutionProvider" in providers:
        badge = "ROCm (GPU)"
        detailed = f"GPU: {gpu_name} / ROCm"
        return badge, detailed, True
    else:
        badge = "CPU Mode"
        detailed = "GPU acceleration unavailable — CPU mode"
        return badge, detailed, False
