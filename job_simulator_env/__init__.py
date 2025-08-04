from gymnasium.envs.registration import register

register(
    id="job-simulator-env/GridWorld-v0",
    entry_point="job-simulator-env.envs:GridWorldEnv",
)
