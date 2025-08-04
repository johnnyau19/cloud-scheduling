import math
from job_simulator_env.envs.config import MAX_ESTIMATED_WORKLOAD, ACTUAL_WORKLOAD_NOISE_STD


class Job():
    def __init__(self, estimated_workload, rng, arrival_time):
        self.estimated_workload = estimated_workload
        self.actual_workload = int(
            round(rng.normal(loc=self.estimated_workload, scale=ACTUAL_WORKLOAD_NOISE_STD)))
        self.actual_workload = max(
            1, min(self.actual_workload, int(math.ceil(1.5 * MAX_ESTIMATED_WORKLOAD))))

        # Arrival time
        self.arrival_time = arrival_time
    
