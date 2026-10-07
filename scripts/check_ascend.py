"""Check the Ascend runtime before launching Qwen training."""
import os
import sys


def main():
    try:
        import torch
        import torch_npu  # noqa: F401 - registers the NPU backend
    except ImportError as exc:
        print(f"Missing Ascend PyTorch dependency: {exc}", file=sys.stderr)
        return 2

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("torch_npu is installed but no NPU is available", file=sys.stderr)
        return 3

    count = torch.npu.device_count()
    print(f"torch={torch.__version__}")
    print(f"npu_count={count}")
    print(f"visible_devices={os.getenv('ASCEND_RT_VISIBLE_DEVICES', '<all>')}")
    for index in range(count):
        torch.npu.set_device(index)
        print(f"npu:{index}={torch.npu.get_device_name(index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
