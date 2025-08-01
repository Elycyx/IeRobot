from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.core.prims import Articulation
from typing import Optional


class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        #TODO: change the config path
        self._kinematics = LulaKinematicsSolver(robot_description_path="/mnt/sda1/lerobot/piper/robot_descriptor.yaml",
                                                urdf_path="/home/cyx/piper/piper.urdf")
        if end_effector_frame_name is None:
            end_effector_frame_name = "link6"
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        return