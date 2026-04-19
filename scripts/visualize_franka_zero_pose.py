from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    xml_path = (
        repo_root
        / "third_party"
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "panda.xml"
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # URDF/MJCF zero pose: qpos = 0 for all joints.
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    print("Franka zero-pose viewer")
    print(f"XML: {xml_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("qpos (zero pose):", np.array(data.qpos).tolist())
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Keep the model at zero pose for direct comparison.
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            if model.nu > 0:
                data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
