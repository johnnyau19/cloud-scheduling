"""
This compares the throughput of RL agent with EEFT policy, and displays the result at the terminal
"""
from job_simulator_env.envs.simulator import JobSimulator
from baseline.eeft import eeft_policy
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import time, random 


# Plot these datapoints onto the graph

# Y axis
agent_throughput = []
eeft_throughput = []

# X axis
episodes = []

# Set up real-time graphing
plt.ion()
fig, ax = plt.subplots()
agent_line, = ax.plot(episodes, agent_throughput, 'r-', label="Agent")
eeft_line, = ax.plot(episodes, eeft_throughput, 'b-', label="EEFT")
ax.set_ylabel("Throughput")
ax.set_xlabel("Episodes")
plt.grid()
plt.legend()
plt.show()

# average throughput of each algorithm
agent_throughput_sum = 0
eeft_throughput_sum = 0

# Initialize the agent
agent = DQN.load("model/best_model")
# agent = DQN.load("logs/checkpoints/dqn_2000000_steps")


# Initalize 2 identical environments, one for RL agent and another for EEFT policy
agent_env = JobSimulator(None)
eeft_env = JobSimulator(None)

test_set = 0
num_test = 2000
while test_set < num_test:
    # Reset the environment with the same seed for both so that the environment assigns similar jobs to both algorithm
    seed = int(time.time_ns())   # Randomly generate test set for each episode
    eeft_obs, info = eeft_env.reset(seed)
    agent_obs, info = agent_env.reset(seed)

    while True:
        # EEFT takes action
        eeft_action = eeft_policy(eeft_obs)
        eeft_obs, eeft_reward, eeft_terminated, eeft_trunc, eeft_info = eeft_env.step(
            eeft_action)

        # Agent takes action
        agent_action, info = agent.predict(agent_obs)
        agent_obs, agent_reward, agent_terminated, agent_trunc, agent_info = agent_env.step(
            int(agent_action))

        if eeft_terminated:
            break

    # Sum the throughtputs to calculate the avg later
    agent_throughput_sum += agent_env.get_total_completed_jobs()
    eeft_throughput_sum += eeft_env.get_total_completed_jobs()

    # Update episode
    episodes.append(test_set)

    # Update the throughput datapoint
    agent_throughput.append(agent_env.get_total_completed_jobs())
    agent_line.set_data(episodes, agent_throughput)

    eeft_throughput.append(eeft_env.get_total_completed_jobs())
    eeft_line.set_data(episodes, eeft_throughput)

    ax.set_xlim(0, max(10, len(episodes)))
    ax.set_ylim(min(agent_throughput + eeft_throughput)-20,
                max(agent_throughput + eeft_throughput) + 20)

    plt.pause(0.001)

    # Display the result
    print(
        f"EEFT policy finished total jobs : {eeft_env.get_total_completed_jobs()} \n")
    print(
        f"Agent finished total jobs : {agent_env.get_total_completed_jobs()} \n")

    # Change another
    test_set += 1
print(f"\nEEFT policy average throughput : {eeft_throughput_sum/num_test} \n")
print(f"Agent average throughput : {agent_throughput_sum/num_test} \n")
print("EEFT won !") if (eeft_throughput_sum/num_test) >= (agent_throughput_sum/num_test) else print("Agent won !")

plt.figure()
plt.ioff()
algorithms= ['DQN Agent', 'EEFT']
throughputs=[agent_throughput_sum/num_test, eeft_throughput_sum/num_test]
plt.bar(algorithms, throughputs, width=0.5)
plt.ylim(100,115)
plt.ylabel('Average throughput per episode')
plt.show()
