from job_simulator_env.envs.simulator import JobSimulator
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Initialize the environment
env =  JobSimulator(None)

# Initialize model
model = DQN(policy="MlpPolicy", env=env, verbose=1)

# Store the state of the model after a log_interval timesteps
checkpoint_callback = CheckpointCallback(save_freq=2000,
                                         save_path="./logs/checkpoints/",
                                         name_prefix="dqn")

# Set up environment for the model
model.set_env(env)

# Train model
model.learn(total_timesteps=2000000, log_interval=10, progress_bar=True, callback=checkpoint_callback)
model.save("model/dqn_job_scheduler")



