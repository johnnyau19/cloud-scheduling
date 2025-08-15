"""
This compares the throughput of RL agent with EEFT policy, and displays the result at the terminal
"""
from job_simulator_env.envs.simulator import JobSimulator
from baseline.eeft import eeft_policy
import matplotlib.pyplot as plt
import time
import onnxruntime as ort


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

# number of times assigned jobs to a full server
num_assign_jobs_to_full_server = 0 

# Initialize the agent
ort_sess = ort.InferenceSession("model/best_model.onnx")

# Initalize 2 identical environments, one for RL agent and another for EEFT policy
agent_env = JobSimulator(None)
eeft_env = JobSimulator(None)

test_set = 0
num_test = 1000
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
        agent_obs = agent_obs.reshape(1,-1)
        agent_action = ort_sess.run(None, {"input": agent_obs})[0]

        # agent_action, info = agent.predict(agent_obs, deterministic=True)
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

    # Change another test set
    test_set += 1

    # Update the number of times agent assigning to a full server
    num_assign_jobs_to_full_server += agent_env.get_num_selected_not_enough_capacity_server()

agent_throughput_avg = agent_throughput_sum/num_test
eeft_throughput_avg = eeft_throughput_sum/num_test

print(f"\nEEFT policy average throughput : {eeft_throughput_avg} \n")
print(f"Agent average throughput : {agent_throughput_avg} \n")
print("EEFT won !") if (eeft_throughput_avg) >= (agent_throughput_avg) else print("Agent won !"), print("Number of times DQN agent assigned jobs to a full server: ", num_assign_jobs_to_full_server)
plt.figure()
plt.ioff()
algorithms= ['DQN Agent', 'EEFT']
throughputs=[agent_throughput_avg, eeft_throughput_avg]
plt.bar(algorithms, throughputs, width=0.5)
plt.ylim(22,max(agent_throughput_avg, eeft_throughput_avg)+0.2)
plt.ylabel('Average throughput per episode')
plt.show()
