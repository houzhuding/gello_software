import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin


def _resolve_default_urdf(repo_root: Path) -> Path:
    return repo_root / "gello" / "factr" / "urdf" / "factr_teleop_franka.urdf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize GELLO Franka leader URDF in Meshcat"
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to URDF (default: gello/factr/urdf/factr_teleop_franka.urdf)",
    )
    parser.add_argument(
        "--q",
        type=float,
        nargs="*",
        default=None,
        help="Optional joint configuration in radians (length must equal nq)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="If >0, auto-exit after this many seconds; otherwise run until Ctrl+C.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    urdf_path = Path(args.urdf) if args.urdf else _resolve_default_urdf(repo_root)
    urdf_path = urdf_path.resolve()

    if not urdf_path.exists():
        print(f"URDF not found: {urdf_path}")
        return 1

    package_dir = urdf_path.parent
    print(f"Loading leader URDF: {urdf_path}")

    try:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            str(urdf_path), package_dirs=str(package_dir)
        )
    except Exception as e:
        print(f"Failed to load URDF: {e}")
        return 1

    try:
        from pinocchio.visualize import MeshcatVisualizer
    except Exception:
        print("Meshcat visualizer is unavailable.")
        print("Install with: pip install meshcat")
        return 1

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True)
        viz.loadViewerModel("gello_franka_leader")
    except Exception as e:
        print(f"Failed to initialize Meshcat viewer: {e}")
        print("If Meshcat is missing, run: pip install meshcat")
        return 1

    q0 = pin.neutral(model)
    if args.q is not None and len(args.q) > 0:
        if len(args.q) != len(q0):
            print(
                f"Invalid --q length {len(args.q)}; expected {len(q0)} (nq). Using neutral."
            )
        else:
            q0 = np.array(args.q, dtype=float)

    print(f"nq={len(q0)}")
    print("Zero pose (neutral q):", [float(f"{x:.6f}") for x in pin.neutral(model).tolist()])
    print("Displayed q:", [float(f"{x:.6f}") for x in q0.tolist()])
    print("A browser tab should open with the leader URDF.")

    viz.display(q0)

    if args.hold_seconds > 0:
        time.sleep(args.hold_seconds)
        return 0

    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
