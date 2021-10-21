
from gym.spaces import Box
from gym import Wrapper
import cv2
from gym_electric_motor import reference_generators
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp
from gym.logger import info
import gym_electric_motor as gem
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from gym_electric_motor import physical_systems as ps
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator
import numpy as np
from gym_electric_motor.core import Callback


def create_train_env():

    motor_env_id = "DcSeriesDisc-v1"

    tau = 1e-5    # The duration of each sampling step
    u_sup = 250
    omega = 33
    current = 100

    referencegen = rg.ConstReferenceGenerator()

    visualization = MotorDashboard(
        state_plots=[
            "omega",
            "torque",
            "i",
            "u",
        ],  # Pass a list of state names or 'all' for all states to plot
        reward_plot=True,  # True / False (False default)
        style="seaborn-darkgrid",
    )

    nominal_values=dict(omega=omega,
                        i=current,
                        u=u_sup
                        )
    limit_values=dict(omega=omega,
                        i=1.5*current,
                        u=u_sup
                        )


    env = gem.make(
        # Define the series DC motor with continuous-control-set
        'DcSeriesDisc-v1',
        # Pass a class with extra parameters
        visualization=MotorDashboard(state_plots=['omega', 'torque', 'i', 'u', 'u_sup'], reward_plot=True),


        # Set the parameters of the motor
        motor_parameter=dict(r_a=2.5, r_e=4.5, l_a=9.7e-3, l_e_prime=9.2e-3, l_e=9.2e-3, j_rotor=0.001),

        # Set the parameters of the mechanical polynomial load (the default load class)
        load_parameter=dict(a=0, b=.0, c=0.01, j_load=.001),

        # Weighting of different addends of the reward function
        # This can be used to assign higher or lower priorities to different states
        # As only one state is weighted here, the value does not make a major difference
        reward_weights={'omega': 1},

        # Exponent of the reward function
        # Here we use a square root function
        # reward_power=0.5,

        # Define which state variables are to be monitored concerning limit violations
        # () means that a limit violation will never trigger an env.reset
        # constraints=(),

        # Define which numerical solver is to be used for the simulation
        ode_solver='scipy.solve_ivp',
        solver_kwargs=dict(method='BDF'),

        # Define and parameterize the reference generator for the speed reference
        # reference_generator=WienerProcessReferenceGenerator(reference_state='omega', sigma_range=(5e-3, 1e-2)),
        reference_generators = rg.ConstReferenceGenerator()
    )


    # env = gem.make(
    #     motor_env_id,
    #     visualization=visualization,
    #     limit_values=limit_values, nominal_values=nominal_values,
    #     reference_generators = referencegen,
    #     tau=tau,
        
    # )


    return env


class MultipleEnvironments:
    def __init__(self, num_envs):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.envs = [create_train_env() for _ in range(num_envs)]
        self.num_states = 2
        self.num_actions = 2
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action.item()))
                
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError