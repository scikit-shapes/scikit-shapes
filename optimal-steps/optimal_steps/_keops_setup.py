"""KeOps compiler setup on macOS, applied when optimal_steps is imported (before pykeops)."""

import os
import sys


def _macos_keops_compiler_fix():
    """Let KeOps compile its CPU kernels with Apple clang.

    KeOps reuses PyTorch's build flags, which contain GCC-only options
    (-march=nocona, ...) and, depending on the toolchain, no path to the macOS
    SDK ("'cmath' file not found"). We point CC/CXX to small wrappers that drop
    those flags and add -isysroot. No-op on Linux, or if CC/CXX are already set.
    """
    if sys.platform != "darwin" or "CXX" in os.environ:
        return
    import stat
    import subprocess
    import tempfile

    try:
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        wrapper_dir = os.path.join(
            tempfile.gettempdir(), f"keops_clang_wrapper_{os.getuid()}"
        )
        os.makedirs(wrapper_dir, exist_ok=True)
        for name, env in (("c++", "CXX"), ("cc", "CC")):
            real = subprocess.check_output(["xcrun", "-f", name], text=True).strip()
            path = os.path.join(wrapper_dir, name)
            with open(path, "w") as f:
                f.write(
                    '#!/bin/bash\nargs=()\nfor arg in "$@"; do\n'
                    '    [[ "$arg" == -march=* || "$arg" == -mtune=* || "$arg" == -mfpmath=* ]] && continue\n'
                    '    args+=("$arg")\ndone\n'
                    f'exec "{real}" -isysroot "{sdk}" "${{args[@]}}"\n'
                )
            os.chmod(
                path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
            os.environ[env] = path
    except Exception:
        pass


_macos_keops_compiler_fix()
