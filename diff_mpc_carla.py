from jax import jit, jacfwd, jacrev, hessian, lax
from jax.scipy.special import logsumexp
import jax
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



np.set_printoptions(precision=3)

import numpy as onp

import pickle
import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = [5, 5]

from tqdm.auto import tqdm

import time

from car_env_for_MPC import *

import os

torch.cuda.is_available()

DT = 0.09# [s] delta time step, = 1/FPS_in_server
TIMESTEPS = 3
# MODEL_NAME = "bicycle_model_100000_v2_jax"
MODEL_NAME = "bicycle_model_100ms_20000_v4_jax"
model_path="./SystemID/model/net_{}.model".format(MODEL_NAME)
NN_W1, NN_W2, NN_W3, NN_LR_MEAN = pickle.load(open(model_path, mode="rb"))

NN_W1 = torch.tensor(NN_W1)
NN_W2= torch.tensor(NN_W2)
NN_W3 = torch.tensor(NN_W3)

NN_LR_MEAN = torch.tensor(NN_LR_MEAN)

# NOTE: Set dp to be the same as carla
dp = 1 # same as waypoint interval

onp.random.seed(1)

TIME_STEPS_RATIO = TIMESTEPS/50
# TARGET_RATIO = np.linalg.norm(target[-1]-target[0])/(3*np.pi)
TARGET_RATIO = FUTURE_WAYPOINTS_AS_STATE*dp/(6*np.pi) # TODO: decide if this should be determined dynamically

if __name__ == "__main__":

    class CarlaDynamics(torch.nn.Module):

        def NN3(self, x):

            x = torch.tanh(torch.mm(NN_W1, x))
            x = torch.tanh(torch.mm(NN_W2, x))
            x = torch.mm(NN_W3, x)
            return x

        def forward(self, state, u):

                # Here we are using Torch Variables ########

                start = datetime.datetime.now()
                
                x = state[:, 0].view(-1, 1)
                y = state[:, 1].view(-1, 1)
                v = state[:, 2].view(-1, 1)

                v_sqrt = torch.sqrt(v)
                phi = state[:, 3].view(-1, 1)
                beta = state[:, 4].view(-1, 1)
                v_sqrt = torch.sqrt(v)

                steering = torch.sin(u[:, 0].view(-1, 1))
                throttle = torch.sin(u[:, 1].view(-1, 1))*0.5 + 0.5
                brake = torch.sin(u[:, 2].view(-1, 1))*0.5 + 0.5

                deriv_x = v*torch.cos(phi+beta)
                deriv_y = v*torch.sin(phi+beta)
                deriv_phi = v*torch.sin(beta)/NN_LR_MEAN
                # Until here ################################

                # Here we need to work with numpy, because x1 and x2 need be numpy vars
                # So I think we use the conevrsion to numpy here (There is some way)
                # Maybe need to use detach
                x1 = torch.cat((
                            torch.sqrt(v),
                            torch.cos(beta),
                            torch.sin(v),
                            steering,
                            throttle,
                            brake
                        ))

                x2= torch.cat((
                            torch.sqrt(v),
                            torch.cos(beta),
                            -torch.sin(v),
                            -steering,
                            throttle,
                            brake
                        ))
                
                # Here we convert x1 and x2 to Pytorch
                x1 = self.NN3(x1)
                x2 = self.NN3(x2)


                # Here continue with the Pytorch Normally
                deriv_v = ( x1[0]*(2*v_sqrt+x1[0]) + x2[0]*(2*v_sqrt+x2[0]) )/2 # x1[0]+x2[0]
                deriv_beta = ( x1[1] - x2[1] )/2

                # deriv_v = torch.from_numpy(deriv_v)
                # deriv_beta = torch.from_numpy(deriv_beta)

                # I change the code to compute the next step inside, directly
                state = torch.cat((x + deriv_x * DT, y + deriv_y * DT, v + deriv_v , phi + deriv_phi * DT, beta + deriv_beta), dim = 1)
                state = state.type(torch.FloatTensor)

                end = datetime.datetime.now()

                return state 


    # carla init
    env = CarEnv()

    # Number of states - 6, number of controlls 3 (steering, throttle and brake)
    n_states = 5
    n_controls = 3
    n_batch = 1
    LQR_ITER = 3

    u_init = None
    render = True

    # Define the costs here

    goal_weights = torch.tensor((5., 5., 0.5, 30, 0.5))  # nx
    ctrl_penalty = 0.001
    goal_state = torch.tensor((20., 20., 0, 0, 0))
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(n_controls)))

    q = torch.cat((
                   goal_weights,
                   ctrl_penalty * torch.ones(n_controls)
                      ))  # nx + nu
    Q = torch.diag(q).repeat(TIMESTEPS, n_batch, 1, 1)  # T x B x nx+nu x nx+nu

    cost = mpc.QuadCost(Q, p) 

    u_lower = -torch.rand(TIMESTEPS, n_batch, n_controls)
    u_upper = torch.rand(TIMESTEPS, n_batch, n_controls)
    
    # Lower constraints
    Lf = 2.67
    for i in range(TIMESTEPS):
        u_lower[i][0][0] = -0.436332
        u_lower[i][0][1] = 0
        u_lower[i][0][2] = -1

    # Upper constraints (Look in your code, it was wrong)
    for i in range(TIMESTEPS):
        u_upper[i][0][0] = 0.436332 
        u_upper[i][0][1] = 1
        u_upper[i][0][2] = 1

    for i in range(1):
        state, waypoints = env.reset()

        # total_time = 0

        for k in tqdm(range(100)):

            state[2] += 0.01
            waypoints = np.array(waypoints)
            state = torch.tensor(state).view(1, -1)
            state = state.type(torch.FloatTensor)

            # p = torch.cat((torch.zeros(n_states), torch.zeros(n_controls)))
            # goal_weights = torch.tensor((0.08, 0.08, 0.05, 0.05, 0.05, 0, 0, 0))
            # goal_state = torch.cat((torch.zeros(n_states), torch.zeros(n_controls)))
            # goal_state[0] = waypoints[TIMESTEPS - 1][0]
            # goal_state[1] = waypoints[TIMESTEPS - 1][1]

            # p = p.repeat(TIMESTEPS, n_batch, 1)

            # for pt_index in range(TIMESTEPS):
            #     p[pt_index][0][0] = waypoints[pt_index][0]
            #     p[pt_index][0][1] = waypoints[pt_index][1]
            #     p[pt_index][0][2] = -15 # 50 kmph is the reference speed
            #     #p[pt_index][0][1] = v_ref
            
            # #p[-1][0] = -torch.sqrt(goal_weights) * goal_state

            # cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

            ctrl = mpc.MPC(n_states, 
                           n_controls, 
                           TIMESTEPS, 
                           u_lower=u_lower, 
                           u_upper=u_upper, 
                           lqr_iter=LQR_ITER,
                           exit_unconverged=False, 
                           eps=1e-2,
                           n_batch=n_batch, 
                           backprop=True, 
                           verbose=3, 
                           u_init=u_init,
                           grad_method=mpc.GradMethods.AUTO_DIFF)

            # compute action based on current state, dynamics, and cost
            start = datetime.datetime.now()
            nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, CarlaDynamics())
            end = datetime.datetime.now()
  
            action = nominal_actions[0]  # take first planned action
            u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, n_controls)), dim=0)
            x_trj = nominal_states.detach().numpy()
            u_trj = nominal_actions
            u_trj = u_trj.detach().numpy()


            start = datetime.datetime.now()
            draw_planned_trj(env.world, x_trj, env.location_[2], color=(0, 223, 222))
            end = datetime.datetime.now()
            
            for j in range(TIMESTEPS):
                steering = np.sin(u_trj[j, 0, 0])
                throttle = np.sin(u_trj[j, 0, 1])*0.5 + 0.5
                brake = np.sin(u_trj[j, 0, 2])*0.5 + 0.5

                start = datetime.datetime.now()
                state, waypoints, done, _ = env.step(onp.array([steering, throttle, brake]))
                end = datetime.datetime.now()
            end = datetime.datetime.now()

    pygame.quit()

    if VIDEO_RECORD:
        os.system("ffmpeg -r 50 -f image2 -i Snaps/%05d.png -s {}x{} -aspect 16:9 -vcodec libx264 -crf 25 -y Videos/result.avi".
        format(RES_X, RES_Y))
