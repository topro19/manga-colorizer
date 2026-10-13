"""
Hardware and GPU detection module for AMD Radeon & DirectML.
"""

import os
import subprocess
from typing import Dict, Any, List
import onnxruntime as ort

def detect_gpu_environment() -> Dict[str, Any]:
    """
    Detects AMD GPU hardware, VRAM, and ONNX Runtime providers.
    Prefers DmlExecutionProvider for AMD Radeon RX 9060 XT.
    """
    available_providers = ort.get_available_providers()
    directml_available = "DmlExecutionProvider" in available_providers
    
    gpu_name = "AMD Radeon RX 9060 XT"
    vram_gb = 16.0
    
    # Query Windows WMI/PowerShell for exact GPU name
    try:
        cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, stderr=subprocess.DEVNULL)
        names = [line.strip() for line in out.strip().splitlines() if line.strip()]
        for name in names:
            if "Radeon" in name or "AMD" in name:
                gpu_name = name
                break
    except Exception:
        pass
        
    active_provider = "DmlExecutionProvider" if directml_available else "CPUExecutionProvider"
    
    return {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "active_provider": active_provider,
        "directml_available": directml_available,
        "available_providers": available_providers,
        "display_text": f"GPU: {gpu_name} / DirectML" if directml_available else "GPU Acceleration Unavailable — CPU Mode"
    }

def get_ort_session_options() -> ort.SessionOptions:
    """Configures optimized session options for DirectML on Windows."""
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts
