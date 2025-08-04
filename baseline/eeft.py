"""
This defines the implementation of EEFT algorithm, this algorithm serves as a competant against the RL agent in the provided environments, both of them attempts to 
maximize the throughput.
"""

from job_simulator_env.envs.config import NUM_SERVERS, SERVER_MAX_WORKLOAD
import numpy as np 


def eeft_policy(observation : np.ndarray) -> int:
    #Convert np array to list 
    observation = observation.tolist()

    #Check the number of servers
    job_estimated_workload = observation.pop()
    servers_est_workload = observation
    
    smallest_workload = min(servers_est_workload)
    if (SERVER_MAX_WORKLOAD-smallest_workload) >= job_estimated_workload:
        # Find the server that has the smallest estimated workload with capacity >= job estimated workload
        for server_id in range(NUM_SERVERS):
            if servers_est_workload[server_id] == smallest_workload:
                return server_id
            
    return 3 #deferral


        
            