"""Script demonstrating PID tracking of a centered horizontal figure-8.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ cd gym-pybullet-drones
    $ python gym_pybullet_drones/examples/figure8_hover.py

Notes
-----
The drone tracks a horizontal figure-8 centered around the world Z axis
(XY center at approximately (0, 0)) while maintaining a fixed altitude.
"""

import argparse
import time

import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 15
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

DEFAULT_FIGURE8_AMPLITUDE = 0.3
DEFAULT_FIGURE8_PERIOD_SEC = 10
DEFAULT_YAW_FOLLOW = False


def run(
    drone=DEFAULT_DRONE,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VISION,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    obstacles=DEFAULT_OBSTACLES,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,
    figure8_amplitude=DEFAULT_FIGURE8_AMPLITUDE,
    figure8_period_sec=DEFAULT_FIGURE8_PERIOD_SEC,
    yaw_follow=DEFAULT_YAW_FOLLOW,
):
    if num_drones != 1:
        raise ValueError("figure8_hover.py currently supports one drone only (use --num_drones 1)")

    #### Initialize the simulation #############################
    init_xyzs = np.array([[0.0, 0.0, 0.6]])
    init_rpys = np.array([[0.0, 0.0, 0.0]])

    #### Initialize a centered horizontal figure-8 trajectory ##
    period = float(figure8_period_sec)
    amplitude = float(figure8_amplitude)
    if period <= 0.0:
        raise ValueError("figure8_period_sec must be > 0")
    if amplitude <= 0.0:
        raise ValueError("figure8_amplitude must be > 0")

    num_wp = int(control_freq_hz * period)
    if num_wp <= 0:
        raise ValueError("figure8_period_sec and control_freq_hz must define at least one waypoint")

    target_pos = np.zeros((num_wp, 3))
    target_rpy = np.zeros((num_wp, 3))
    target_z = init_xyzs[0, 2]
    omega = (2.0 * np.pi) / period

    """
    Figure 8 trajectory
    for i in range(num_wp):
        theta = 2.0 * np.pi * (i / num_wp)
        # Lissajous-style figure-8 centered at (0, 0)
        x = amplitude * np.sin(theta)
        y = 0.5 * amplitude * np.sin(2.0 * theta)
        target_pos[i, :] = np.array([x, y, target_z])
        if yaw_follow:
            dx_dt = amplitude * np.cos(theta) * omega
            dy_dt = amplitude * np.cos(2.0 * theta) * omega
            target_rpy[i, 2] = np.arctan2(dy_dt, dx_dt)
    """

    #Rose curve with n amount of petals, centered at (0, 0)
    petals = 4
    for i in range(num_wp):
        theta = 2.0 * np.pi * (i / num_wp)
        r = amplitude * np.cos(petals * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        target_pos[i, :] = np.array([x, y, target_z])
        if yaw_follow:
            dr_dt = -amplitude * petals * np.sin(petals * theta)
            dx_dt = dr_dt * np.cos(theta) - r * np.sin(theta)
            dy_dt = dr_dt * np.sin(theta) + r * np.cos(theta)
            target_rpy[i, 2] = np.arctan2(dy_dt, dx_dt)
    
    wp_counter = 0

    #### Create the environment ################################
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui,
    )

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )

    #### Initialize the controller #############################
    if drone not in [DroneModel.CF2X, DroneModel.CF2P]:
        raise ValueError("DSLPIDControl supports only cf2x/cf2p")
    ctrl = [DSLPIDControl(drone_model=drone)]

    #### Run the simulation ####################################
    action = np.ones((num_drones, 4)) * env.HOVER_RPM
    start = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        target_idx = wp_counter
        target_rpy_cmd = target_rpy[target_idx, :] if yaw_follow else init_rpys[0, :]

        action[0, :], _, _ = ctrl[0].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos[target_idx, :],
            target_rpy=target_rpy_cmd,
        )

        wp_counter = wp_counter + 1 if wp_counter < (num_wp - 1) else 0

        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([target_pos[target_idx, :], target_rpy_cmd, np.zeros(6)]),
        )

        env.render()
        if gui:
            sync(i, start, env.CTRL_TIMESTEP)

    #### Close and save ########################################
    env.close()
    logger.save()
    logger.save_as_csv("figure8_hover")

    #### Plot ###################################################
    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Figure-8 hover script using CtrlAviary and DSLPIDControl"
    )
    parser.add_argument(
        "--drone",
        default=DEFAULT_DRONE,
        type=DroneModel,
        help="Drone model (default: CF2X)",
        metavar="",
        choices=DroneModel,
    )
    parser.add_argument(
        "--num_drones",
        default=DEFAULT_NUM_DRONES,
        type=int,
        help="Number of drones (default: 1)",
        metavar="",
    )
    parser.add_argument(
        "--physics",
        default=DEFAULT_PHYSICS,
        type=Physics,
        help="Physics updates (default: PYB)",
        metavar="",
        choices=Physics,
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=DEFAULT_RECORD_VISION,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--plot",
        default=DEFAULT_PLOT,
        type=str2bool,
        help="Whether to plot simulation results (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--user_debug_gui",
        default=DEFAULT_USER_DEBUG_GUI,
        type=str2bool,
        help="Whether to add debug lines and GUI sliders (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--obstacles",
        default=DEFAULT_OBSTACLES,
        type=str2bool,
        help="Whether to add obstacles to the environment (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--simulation_freq_hz",
        default=DEFAULT_SIMULATION_FREQ_HZ,
        type=int,
        help="Simulation frequency in Hz (default: 240)",
        metavar="",
    )
    parser.add_argument(
        "--control_freq_hz",
        default=DEFAULT_CONTROL_FREQ_HZ,
        type=int,
        help="Control frequency in Hz (default: 48)",
        metavar="",
    )
    parser.add_argument(
        "--duration_sec",
        default=DEFAULT_DURATION_SEC,
        type=int,
        help="Duration of simulation in seconds (default: 15)",
        metavar="",
    )
    parser.add_argument(
        "--figure8_amplitude",
        default=DEFAULT_FIGURE8_AMPLITUDE,
        type=float,
        help="Figure-8 amplitude in meters (default: 0.3)",
        metavar="",
    )
    parser.add_argument(
        "--figure8_period_sec",
        default=DEFAULT_FIGURE8_PERIOD_SEC,
        type=float,
        help="Figure-8 period in seconds (default: 10)",
        metavar="",
    )
    parser.add_argument(
        "--yaw_follow",
        default=DEFAULT_YAW_FOLLOW,
        type=str2bool,
        help="Whether yaw should align with figure-8 tangent (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    args = parser.parse_args()

    run(**vars(args))
