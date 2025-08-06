from job_simulator_env.envs.simulator import JobSimulator
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import time

# Initialize the environment
train_env =  JobSimulator(None)
val_env =  JobSimulator(None)


# Initialize model
model = DQN(policy="MlpPolicy", env=train_env, verbose=1)

# Store the state of the model after a log_interval timesteps
# checkpoint_callback = CheckpointCallback(save_freq=2000,
#                                          save_path="./logs/checkpoints/",
#                                          name_prefix="dqn")

# Save the checkpoint of the model the agent achieves the highest mean rew during training
eval_callback = EvalCallback(n_eval_episodes=50,eval_env=val_env, eval_freq=20000, best_model_save_path="logs/best_model_checkpoint")

# Set up environment for the model
model.set_env(train_env)

# Train model   
model.learn(total_timesteps=1800000, log_interval=50, progress_bar=True, callback=eval_callback)
model.save("model/dqn_job_scheduler")



