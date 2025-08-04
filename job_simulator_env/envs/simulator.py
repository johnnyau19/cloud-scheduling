from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from job_simulator_env.envs.config import *
from job_simulator_env.envs.job import Job
import math


class Actions(Enum):
    server_1 = 0
    server_2 = 1
    server_3 = 2
    deferral = 3


class JobSimulator(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human"):
        self.num_servers = NUM_SERVERS
        self.server_max_workload = SERVER_MAX_WORKLOAD
        self.max_estimated_workload = MAX_ESTIMATED_WORKLOAD

        # Create a queue for each server
        self.servers_queue = [deque() for _ in range(self.num_servers)]

        self.observation_space = spaces.Box(
            low=0,
            high=self.server_max_workload,
            shape=(self.num_servers + 1,),  # 3 servers + 1 job workload
            dtype=np.float32
        )


        self.action_space = spaces.Discrete(self.num_servers + 1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        job_workload = self.agent_holding_job.estimated_workload if self.agent_holding_job else 0.0
        return np.concatenate((
            self.servers_estimated_workload.astype(np.float32), 
            np.array([job_workload], dtype=np.float32)
        ))

    def _get_info(self):
        return {}

    def job_generator(self):
        workload = self.np_random.normal(
            loc=ESTIMATED_WORKLOAD_MEAN, scale=ESTIMATED_WORKLOAD_STD)
        workload = int(math.ceil(workload))
        workload = max(1, min(MAX_ESTIMATED_WORKLOAD, workload))
        self.agent_holding_job = Job(
            workload, self.np_random, arrival_time=self.current_step)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_completed_jobs = 0 
        for queue in self.servers_queue:
            queue.clear()
        self.servers_estimated_workload = np.zeros(
            shape=(self.num_servers, ), dtype=np.float32)

        # Create a new job, and assign it to the agent
        self.job_generator()
        self._render_frame(job=self.agent_holding_job)
        return self._get_obs(), self._get_info()  # return observation and info

    def step(self, action):

        # Step increases
        self.current_step += 1

        # Check if this step is terminated
        terminated = (self.current_step >= 200)

        # Reset reward
        reward = 0

        # If agent does not choose to deferral, then we create a new job and assign to the agent
        # If agent deferrals, then simulator will neither generate nor assign new job to the agent
        if action != Actions.deferral.value:
            # Give bonus if it correctly assigns job to the queue has smallest workload
            if self.agent_holding_job.estimated_workload <= (self.server_max_workload - self.servers_estimated_workload[action]):

                # Find the current servers that has the smallest estimated workload
                smallest_workload = min(self.servers_estimated_workload)
                servers_with_min_workload = set()
                for id in range(len(self.servers_estimated_workload)):
                    if self.servers_estimated_workload[id] == smallest_workload:
                        servers_with_min_workload.add(id)

                # Check if the agent 's selected server is one these severs that has the min workload
                if action in servers_with_min_workload:
                    reward += CORRECTLY_SELECTED_MIN_WORKLOAD_SERVER
                else:
                    # If agent selected the wrong server's queue, punish
                    reward += INCORRECTLY_SELECTED_MIN_WORKLOAD_SERVER

            # Punish if the agent assigned job to the server that does not have capacity >= job's estimated workload
            else:
                reward += SELECTED_NOT_ENOUGH_SERVER_CAPACITY

            # Add job to the queue of the server
            self.servers_queue[action].append(self.agent_holding_job)

            # Increase the total estimated workload of the server by the estimated workload of a newly assigned job
            self.servers_estimated_workload[action] += self.agent_holding_job.estimated_workload

            # Agent with no job, since its assigned to a server's queue
            self.agent_holding_job = None

            # If new state is not terminal, then continue generating job and assign to the agent
            self.job_generator() if not terminated else None

            # since agent is not keeping the job, so we turn off
            wait = False

        # If agent chooses to deferral
        elif action == Actions.deferral.value:
            spare_servers = 0
            for workload in self.servers_estimated_workload:
                if self.agent_holding_job.estimated_workload <= (self.server_max_workload-workload):
                    spare_servers += 1
                    break

            # If there is a spare server, there is no point for not assigning the job away
            if spare_servers > 0:
                reward += DEFERRAL_WHILE_SPARE_SERVER_EXISTS

            # No server has capacity for this job, agent should keep the job and wait
            elif spare_servers == 0:
                reward += CORRECTLY_DEFERRAL
            wait = True  # boolean value to show the agent's state when agent is waiting

        # Servers are processing jobs
        for server_id in range(self.num_servers):
            # Check if the server's queue is empty
            if self.servers_queue[server_id]:
                # Simulating job handling by decreasing the actual workload of the job
                self.servers_queue[server_id][0].actual_workload -= PROCESS_SPEED

                # Remove it from the server's queue when actual workload = 0 , give it a reward
                if self.servers_queue[server_id][0].actual_workload <= 0:
                    finished_job = self.servers_queue[server_id].popleft()
                    self.servers_estimated_workload[server_id] -= finished_job.estimated_workload
                    self.total_completed_jobs += 1
                    # reward += 1.7**-((self.current_step-finished_job.arrival_time)-0.599)   #JOB_DONE
                    reward += JOB_DONE*(0.9**-((self.current_step-finished_job.arrival_time)))

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame(assign_job_to_server_id=action,
                           job=self.agent_holding_job, deferral=wait)

        return observation, reward, terminated, False, info
    
    def get_total_completed_jobs(self):
        return self.total_completed_jobs

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self, assign_job_to_server_id: int = None, job: Job = None, deferral: bool = False):
        if self.render_mode == "human":
            print("\n\n\n","="*180)
            # Step
            print("STEP : ", self.current_step)
            print("-"*180, "\n")

            # Print agent's status
            if deferral:
                print(
                    f"(ง •̀_•́)ง   : Holding the job weighing {job.estimated_workload} still !\n")
            else:
                print(
                    f"( •̀_•́)    : Assigned to server {assign_job_to_server_id}\n") if assign_job_to_server_id is not None else None
                print(
                    f"(ง •̀_•́)ง   : Holding a new job weighing {job.estimated_workload} !\n") if job is not None else None

            print("\n")
            # Print servers processing jobs
            for id in range(self.num_servers):
                # Total estimated workload of a server
                print(
                    f"Estimated workload = {self.servers_estimated_workload[id]}")
                # Server id
                print(f"[ Server {id} ]", end=" | ")
                # Print all the estimated workloads (representing jobs)
                for job in self.servers_queue[id]:
                    print(f"{job.estimated_workload}", end=" | ")
                print("\n\n")

            print("="*180, "\n\n\n")

    def close(self):
        pass
