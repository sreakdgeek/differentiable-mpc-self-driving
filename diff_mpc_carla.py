
from mpc import mpc
import logging
import math
import time
import gym
import numpy as np
import torch
import torch.autograd
from gym import wrappers, logger as gym_log
from mpc.mpc import QuadCost, LinDx
import datetime
from tqdm.auto import tqdm
import time
from car_env_for_MPC import *
from mpc_util import *
import os
import pickle
import matplotlib.pyplot as plt


np.set_printoptions(precision=3)

plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = [5, 5]

torch.cuda.is_available()

DT = 0.1 # [s] delta time step, = 1/FPS_in_server
TIMESTEPS = 10
VIDEO_RECORD = True


# NOTE: Set dp to be the same as carla
dp = 1 # same as waypoint interval

np.random.seed(1)

TIME_STEPS_RATIO = TIMESTEPS/50
# TARGET_RATIO = np.linalg.norm(target[-1]-target[0])/(3*np.pi)
TARGET_RATIO = FUTURE_WAYPOINTS_AS_STATE*dp/(6*np.pi) # TODO: decide if this should be determined dynamically


if __name__ == "__main__":

    class CarlaDynamics(torch.nn.Module):

        # Torch ain't got any polyval, what do I do?
        def myPolyVal(self, coeffs, x):
            curVal = 0
            for curValIndex in range(len(coeffs) - 1):
                curVal = (curVal +  coeffs[curValIndex]) * x[0]
            return (curVal + coeffs[len(coeffs) - 1])

        # State - x, y, v, theta (heading angle), CTE (Cross track error), ETH (Heading error)
        def forward(self, init_state, inputs):
            
            init_x = init_state[:, 0].view(-1, 1)
            init_y = init_state[:, 1].view(-1, 1)
            init_v = init_state[:, 2].view(-1, 1)
            init_theta = init_state[:, 3].view(-1, 1)
            init_cte = init_state[:, 4].view(-1, 1)
            init_eth = init_state[:, 5].view(-1, 1)

            steer_angle = inputs[:, 0].view(-1, 1)
            acceleration = inputs[:, 1].view(-1, 1)

            dt = 0.1
            L = 2.67

            ## fcind the final satte after dt of time ###
            final_x  = init_x  + init_v * torch.cos(init_theta) * dt
            final_y  = init_y  + init_v * torch.sin(init_theta) * dt
            final_th = init_theta + (init_v / L) * steer_angle * dt
            final_v  = init_v  + acceleration * dt

            theta_des = torch.atan(coeffs[2] + 2 * coeffs[1] * init_x + 3 * coeffs[0] * init_x**2)
            final_cte = self.myPolyVal(coeffs, init_x) - init_y + (init_v * torch.sin(init_eth) * dt)
            final_eth = init_theta - theta_des + ((init_v/L) * steer_angle * dt)

            final_state = torch.cat((final_x, final_y, final_th, final_v, final_cte, final_eth), dim = 1)
            final_state = final_state.type(torch.FloatTensor)

            return final_state


    # carla init
    env = CarEnv()

    # Number of states - 6, number of controlls 3 (steering, throttle and brake)
    n_states = 6
    n_controls = 3
    n_batch = 1
    LQR_ITER = 20

    u_init = None
    render = True

    steering_list = []
    throttle_list = []
    brake_list = []

    state_x_list = []
    state_y_list = []
    vel_list = []

    mpc_costs = []
    states_list = []


    for i in range(1):

        # Get the state and waypoints from the environment

        state, waypoints = env.reset()
        state = torch.tensor(state).view(1, -1)
        state = state.type(torch.FloatTensor)

        # Define the costs weights
        goal_weights = torch.tensor((1., 1., 1., 1., 1., 1.))  # nx
        control_weights = torch.tensor((1, 1., 1))

        q = torch.cat((
                    goal_weights,
                    control_weights
                        ))  # nx + nu
        Q = torch.diag(q).repeat(TIMESTEPS, n_batch, 1, 1)  # T x B x nx+nu x nx+nu

        u_lower = -torch.rand(TIMESTEPS, n_batch, n_controls)
        u_upper = torch.rand(TIMESTEPS, n_batch, n_controls)
    
        # Lower constraints
        for i in range(TIMESTEPS):
            u_lower[i][0][0] = -0.436332
            u_lower[i][0][1] = 0
            u_lower[i][0][2] = 0

        # Upper constraints
        for i in range(TIMESTEPS):
            u_upper[i][0][0] = 0.436332 
            u_upper[i][0][1] = 1
            u_upper[i][0][2] = 1


        # MPC control for each iteration

        for k in tqdm(range(100)):
            waypoints = np.array(waypoints)
            state = torch.tensor(state).view(1, -1)
            state = state.type(torch.FloatTensor)

            waypoints_veh = map_coord_2_Car_coord(state[:, 0], state[:, 1], state[:, 4], waypoints)
            wps_vehRef_x = waypoints_veh[0,:]
            wps_vehRef_y = waypoints_veh[1,:]
            coeffs = np.polyfit(wps_vehRef_x, wps_vehRef_y, 3)

            # Initially state dimension is 2-D, expand it to 6-D, also get a default action vector
            if k == 0:
                state = initialize_state(state, coeffs)
                u_trj = initialize_action(TIMESTEPS, n_controls)
            
            p = torch.cat((torch.zeros(n_states), torch.zeros(n_controls)))
            p = p.repeat(TIMESTEPS, n_batch, 1)

            # Control loop

            for t in range(TIMESTEPS):
                v_ref = 20
                goal_weights = torch.tensor((0., 0., 1000., 0., 2000., 30000.))
                goal_state = torch.tensor((0., 0., v_ref, 0., 0., 0))
                state_cost = -torch.sqrt(goal_weights) * goal_state
                control_action = torch.tensor((0, 0, 0))
                control_weights = torch.tensor((100., 1., 0.05))
                control_cost = -torch.sqrt(control_weights) * control_action
                iteration_cost = torch.cat((state_cost, control_cost))
                p[t][0] = iteration_cost

            total_cost = mpc.QuadCost(Q, p) 

            # Set up optimization problem
            ctrl = mpc.MPC(n_states, 
                           n_controls, 
                           TIMESTEPS, 
                           u_lower=u_lower, 
                           u_upper=u_upper, 
                           lqr_iter=LQR_ITER,
                           exit_unconverged=False, 
                           n_batch=n_batch, 
                           backprop=False, 
                           verbose=2, 
                           u_init=u_init,
                           slew_rate_penalty=True,
                           grad_method=mpc.GradMethods.AUTO_DIFF)

            # compute action based on current state, dynamics, and cost
            nominal_states, nominal_actions, nominal_objs = ctrl(state, total_cost, CarlaDynamics())
  
            action = nominal_actions[0]  # take first planned action
            u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, n_controls)), dim=0)
            x_trj = nominal_states.detach().numpy()
            x_trj = get_world_coordinates(x_trj, state)
            u_trj = nominal_actions
            u_trj = u_trj.detach().numpy()

            start = datetime.datetime.now()
            draw_planned_trj(env.world, x_trj, env.location_[2], color=(0, 223, 222))
            end = datetime.datetime.now()

            # Update the controls
            
            for j in range(MPC_INTERVAL):
                steering = np.sin(u_trj[j, 0, 0])
                throttle = np.sin(u_trj[j, 0, 1]) * 0.5 + 0.5
                brake = np.sin(u_trj[j, 0, 2]) * 0.5 + 0.5

                steering_list.append(steering)
                throttle_list.append(throttle)
                brake_list.append(brake)

                state_x_list.append(state[:, 0])
                state_y_list.append(state[:, 1])
                vel_list.append(state[:, 2] * 3.6)

            
                #states_list.append([state[:, 0], state[:, 1], state[:, 2]])

                start = datetime.datetime.now()
                state, waypoints, done, _ = env.step(onp.array([steering, throttle, brake]))
                state = torch.tensor(state).view(1, -1)
                state = state.type(torch.FloatTensor)

                # Reinitalize the coeffs as per the new way points
                waypoints_veh = map_coord_2_Car_coord(state[:, 0], state[:, 1], state[:, 4], waypoints)
                wps_vehRef_x = waypoints_veh[0,:]
                wps_vehRef_y = waypoints_veh[1,:]
                coeffs = np.polyfit(wps_vehRef_x, wps_vehRef_y, 3)
                state = initialize_state_action(state, steering, coeffs)

    f = plt.figure()
    f.suptitle('Control Inputs')
    plt.xlabel('Iteration')
    plt.ylabel('Steering - Throttle - Breake')
    plt.plot(steering_list, '-r', label = 'Steering')
    plt.plot(throttle_list, '-g', label = 'Throttle')
    plt.plot(brake_list, '-b', label = 'Brake')
    leg = plt.legend()
    plt.show()
    f.savefig('controls.pdf')

    f = plt.figure()
    f.suptitle('States')
    plt.xlabel('Iteration')
    plt.ylabel('X, Y')
    plt.plot(state_x_list, '-b', label = 'X')
    plt.plot(state_y_list, '-r', label = 'Y')
    leg = plt.legend()
    plt.show()
    f.savefig('states.pdf')
  
                
    pygame.quit()

    if VIDEO_RECORD:
        os.system("ffmpeg -r 50 -f image2 -i Snaps/%05d.png -s {}x{} -aspect 16:9 -vcodec libx264 -crf 25 -y Videos/result.avi".
        format(RES_X, RES_Y))
