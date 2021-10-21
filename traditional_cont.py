from gym.core import ActionWrapper
from gym.logger import warn
from src.env import create_train_env
import gym_electric_motor as gem
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from gym_electric_motor import physical_systems as ps
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
import numpy as np

from simple_controllers import Controller
import wandb 
wandb.init(project= 'Motor_rl')


if __name__ == "__main__":
    motor_env_id = "DcSeriesCont-v1"



   # Default DcSeries Motor Parameters are changed to have more dynamic system and to see faster learning results.
    env = gem.make(
        # Define the series DC motor with continuous-control-set
        motor_env_id,
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

   
    max_steps = 50000
    total_experiment = 100

    for exp in range(total_experiment):
        total_reward = 0
        state, reference = env.reset()
        done = False
        controller = Controller.make('pid_controller',env,k_p= 20, k_i = 12, k_d = 0)
        for step in range(max_steps+1):
            # the render command updates the dashboard
            env.render()

            action = controller.control(state, reference)
        
            # the drive environment accepts the action and simulates until the next time step
            (state, reference), reward, done, _ = env.step(action)
            total_reward = total_reward + reward
        average_reward = total_reward/max_steps
        
        wandb.log({'experiment_ no ':exp,'average_return ':average_reward})

        print('average reward {} for experiment no {}'.format(average_reward,exp))  
        if controller is not None:
            del controller
