from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
import numpy as np

my_world = World(stage_units_in_meters=1.0)

# use Isaac Sim provided asset
asset_path = "/home/cyx/piper/piper/piper.usd"

#TODO: change this to your own path if you downloaded the asset
# asset_path = "/home/user_name/cobotta_pro_900/cobotta_pro_900/cobotta_pro_900.usd"

add_reference_to_stage(usd_path=asset_path, prim_path="/World/piper")
#define the gripper
gripper = ParallelGripper(
    #We chose the following values while inspecting the articulation
    end_effector_prim_path="/World/piper/link6",
    joint_prim_names=["joint7", "joint8"],
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0, 0]),
    action_deltas=np.array([-0.04, -0.04]),
)
#define the manipulator
my_piper = my_world.scene.add(SingleManipulator(prim_path="/World/piper", name="piper_robot",
                                                end_effector_prim_name="link6", gripper=gripper))
#set the default positions of the other gripper joints to be opened so
#that its out of the way of the joints we want to control when gripping an object for instance.
joints_default_positions = np.zeros(8)
joints_default_positions[6] = 0
joints_default_positions[7] = 0
my_piper.set_joints_default_state(positions=joints_default_positions)
my_world.scene.add_default_ground_plane()
my_world.reset()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        i += 1
        gripper_positions = my_piper.gripper.get_joint_positions()
        if i < 500:
            #open the gripper slowly
            my_piper.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + 0.008, gripper_positions[1] + 0.008]))
        if i > 500:
            #close the gripper slowly
            my_piper.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] - 0.008, gripper_positions[1] - 0.008]))
        if i == 1000:
            i = 0

simulation_app.close()