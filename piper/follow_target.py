from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.api.tasks as tasks
from typing import Optional
import numpy as np


# Inheriting from the base class Follow Target
class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "piper_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        #TODO: change this to the robot USD file.
        asset_path = "/home/cyx/piper/piper/piper.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/piper")
        gripper = ParallelGripper(
            #We chose the following values while inspecting the articulation
            end_effector_prim_path="/World/piper/link6",
            joint_prim_names=["joint7", "joint8"],
            joint_opened_positions=np.array([0.04, 0.04]),
            joint_closed_positions=np.array([0, 0]),
            action_deltas=np.array([-0.04, -0.04]),
        )
        manipulator = SingleManipulator(prim_path="/World/piper", name="piper_robot",
                                                end_effector_prim_name="link6", gripper=gripper)
        joints_default_positions = np.zeros(8)
        joints_default_positions[6] = 0
        joints_default_positions[7] = 0
        manipulator.set_joints_default_state(positions=joints_default_positions)
        return manipulator