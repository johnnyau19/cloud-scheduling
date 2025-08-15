from stable_baselines3 import DQN
import onnxruntime as ort
import time
from job_simulator_env.envs.simulator import JobSimulator
import numpy as np
from tqdm import tqdm

# Initialize models
onnx_dqn_model = ort.InferenceSession("model/best_model.onnx")
pytorch_dqn_model = DQN.load("model/best_model")

# Initialize environments
onnx_env = JobSimulator(None)
pytorch_env = JobSimulator(None)

# Number of test
num_test = input("How many test would you like to run: ")


# Find a random set of seeds
seeds = np.random.randint(0, 1_000_000, size=int(num_test))

test_set = 0 
# total time
onnx_runtime = 0
for seed in tqdm(seeds, desc="ONNX runtime", colour="green", ascii=" ━", smoothing=0.8):
   # Reset environment
    onnx_env_obs, onnx_env_info = onnx_env.reset(seed=int(seed))

    while True:
        # Agent takes action
        onnx_env_obs = onnx_env_obs.reshape(1,-1)
        start = time.time_ns()
        action = onnx_dqn_model.run(None, {"input": onnx_env_obs})[0]
        end = time.time_ns()
        onnx_runtime += end-start
        onnx_env_obs, reward, terminated, trunc, info = onnx_env.step(int(action.item()))
        
        # The end of one episode
        if terminated:
            break


test_set = 0 
# total time
pytorch_runtime = 0
# Start timing
for seed in tqdm(seeds, desc="PYTORCH runtime", colour="green", ascii=" ━", smoothing=0.8):
    # Reset environment
    pytorch_env_obs, pytorch_env_info = pytorch_env.reset(seed=int(seed))

    while True:
        # Agent takes action
        start = time.time_ns()
        action, info = pytorch_dqn_model.predict(pytorch_env_obs, deterministic=True)
        end = time.time_ns()
        pytorch_runtime += end-start
        pytorch_env_obs, reward, terminated, trunc, info = pytorch_env.step(int(action.item()))

        # The end of one episode
        if terminated:
            break

print(f"ONNX runtime :    {onnx_runtime/1e6:.2f} ms")
print(f"PYTORCH runtime : {pytorch_runtime/1e6:.2f} ms")    






