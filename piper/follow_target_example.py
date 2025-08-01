from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from follow_target import FollowTarget
from ik_solver import KinematicsSolver
import numpy as np
import carb

my_world = World(stage_units_in_meters=1.0)
#Initialize the Follow Target task with a target location for the cube to be followed by the end effector
my_task = FollowTarget(name="piper_follow_target", target_position=np.array([0.5, 0, 0.5]))
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("piper_follow_target").get_params()
target_name = task_params["target_name"]["value"]
piper_name = task_params["robot_name"]["value"]
my_piper = my_world.scene.get_object(piper_name)

#initialize the controller
my_controller = KinematicsSolver(my_piper)
articulation_controller = my_piper.get_articulation_controller()

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()

        observations = my_world.get_observations()
        actions, succ = my_controller.compute_inverse_kinematics(
            target_position=observations[target_name]["position"],
            target_orientation=observations[target_name]["orientation"],
        )
        if succ:
            articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")
simulation_app.close()