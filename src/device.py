from __future__ import annotations

import mindspore as ms


def configure_device(device_target: str = "auto") -> str:
    target = (device_target or "auto").strip()
    if target.lower() != "auto":
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=target)
        print(f"MindSpore device_target={target}", flush=True)
        return target

    last_error: Exception | None = None
    for candidate in ("Ascend", "GPU", "CPU"):
        try:
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target=candidate)
            print(f"MindSpore device_target={candidate}", flush=True)
            return candidate
        except Exception as exc:  # pragma: no cover - backend-dependent
            last_error = exc

    raise RuntimeError("Could not configure MindSpore device target.") from last_error
