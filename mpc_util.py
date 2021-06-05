import torch
import pickle
import numpy as np

MODEL_NAME = "bicycle_model_100ms_20000_v4_jax"
model_path="./SystemID/model/net_{}.model".format(MODEL_NAME)
NN_W1, NN_W2, NN_W3, NN_LR_MEAN = pickle.load(open(model_path, mode="rb"))

NN_W1 = torch.tensor(NN_W1)
NN_W2= torch.tensor(NN_W2)
NN_W3 = torch.tensor(NN_W3)

NN_LR_MEAN = torch.tensor(NN_LR_MEAN)
TIMESTEPS = 10

def my_NN3(x):

    x = torch.tanh(torch.mm(NN_W1, x))
    x = torch.tanh(torch.mm(NN_W2, x))
    x = torch.mm(NN_W3, x)
    return x

def myPolyVal(coeffs, x):
    curVal = 0
    for curValIndex in range(len(coeffs) - 1):
        curVal = (curVal +  coeffs[curValIndex]) * x[0]
    return (curVal + coeffs[len(coeffs) - 1])

def initialize_state(state, coeffs):

    # State - x, y, v, theta (heading angle), CTE (Cross track error), ETH (Heading error)
    
    x = state[:, 0].view(-1, 1)
    y = state[:, 1].view(-1, 1)
    v = state[:, 2].view(-1, 1)
    steer_angle = 0
    dt = 0.1
    L = 3

    theta = state[:, 4].view(-1, 1)
    theta_des = torch.atan(coeffs[2] + 2 * coeffs[1] * x + 3 * coeffs[0] * x**2)
    eth = theta - theta_des + ((v/L) * steer_angle * dt)
    np_polyval = np.polyval(coeffs, x)
    my_polyval = myPolyVal(coeffs, x)

    cte = myPolyVal(coeffs, x) - y + (v * torch.sin(eth) * dt)

    new_state = torch.cat((x, y, theta, v, cte, eth), dim = 1)

    return new_state


def initialize_state_action(state, steer_angle, coeffs):

    # State - x, y, v, theta (heading angle), CTE (Cross track error), ETH (Heading error)
    
    x = state[:, 0].view(-1, 1)
    y = state[:, 1].view(-1, 1)
    v = state[:, 2].view(-1, 1)

    dt = 0.1
    L = 3

    theta = state[:, 4].view(-1, 1)
    theta_des = torch.atan(coeffs[2] + 2 * coeffs[1] * x + 3 * coeffs[0] * x**2)
    eth = theta - theta_des + ((v/L) * steer_angle * dt)
    cte = myPolyVal(coeffs, x) - y + (v * torch.sin(eth) * dt)

    new_state = torch.cat((x, y, theta, v, cte, eth), dim = 1)

    return new_state


def initialize_action(timesteps, n_controls):

    n_batch = 1
    u_trj = np.random.randn(timesteps, n_batch,  n_controls) * 1e-8
    u_trj[:,0, 2] -= np.pi/2.5
    u_trj = np.array(u_trj)

    return u_trj

def my_dynamics(inputs, init_state, coeffs, dt = 0.1, L = 3):           
            init_x = init_state[:, 0].view(-1, 1)
            init_y = init_state[:, 1].view(-1, 1)
            init_v = init_state[:, 2].view(-1, 1)
            init_theta = init_state[:, 3].view(-1, 1)
            init_cte = init_state[:, 4].view(-1, 1)
            init_eth = init_state[:, 5].view(-1, 1)

            steer_angle = inputs[:, 0].view(-1, 1)
            acceleration = inputs[:, 1].view(-1, 1)

            ## find the final satte after dt of time ###
            final_x  = init_x  + init_v * torch.cos(init_theta) * dt
            final_y  = init_y  + init_v * torch.sin(init_theta) * dt
            final_th = init_theta + (init_v / L) * steer_angle * dt
            final_th = final_th.type(torch.FloatTensor)
            final_v  = init_v  + acceleration * dt
            final_v = final_v.type(torch.FloatTensor)

            theta_des = torch.atan(coeffs[2] + 2 * coeffs[1] * init_x + 3 * coeffs[0] * init_x**2)
            final_cte = myPolyVal(coeffs, init_x) - init_y + (init_v * torch.sin(init_eth) * dt)
            final_eth = init_theta - theta_des + ((init_v/L) * steer_angle * dt)
            final_eth = final_eth.type(torch.FloatTensor)

            final_state = torch.cat((final_x, final_y, final_th, final_v, final_cte, final_eth), dim = 1)
            final_state = final_state.type(torch.FloatTensor)

            return final_state

def get_world_coordinates(x_trj, state):
    for i in range(x_trj.shape[0]):
        x_trj[i, 0, 0] -= state[:, 0]
        x_trj[i, 0, 1] -= state[:, 1]
    return x_trj

def map_coord_2_Car_coord(x, y, yaw, waypoints):  
    
        wps = np.squeeze(waypoints)
        wps = torch.from_numpy(wps)
        wps_x = wps[:,0]
        wps_y = wps[:,1]

        num_wp = wps.shape[0]
        
        ## create the Matrix with 3 vectors for the waypoint x and y coordinates w.r.t. car 
        wp_vehRef = torch.zeros(size=(3, num_wp))
        cos_yaw = torch.cos(-yaw)
        sin_yaw = torch.sin(-yaw)
                

        wp_vehRef[0,:] = cos_yaw * (wps_x - x) - sin_yaw * (wps_y - y)
        wp_vehRef[1,:] = sin_yaw * (wps_x - x) + cos_yaw * (wps_y - y)

        return wp_vehRef