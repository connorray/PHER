import numpy as np


# HYPER PARAMS
DISCOUNT_FACTOR_GAMMA = 0.99
LEARNING_RATE = 0.0001
UPDATE_EVERY = 4
BATCH_SIZE = 64
TARGET_UPDATE_EVERY = 10000
EXPLORE_STEPS = 10000
MEM_SIZE = 50000
# MEM_SIZE = 100000
MAX_STEPS = 5000000  # Montezuma
SAVE_FREQ = 500000
EVAL_EVERY = 100000  # Montezuma
EVAL_STEPS = 20000
# EVAL_STEPS = 200000  # for visual
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_STEPS = 100000
LOG_EVERY = 10000
SIDE_BOXES = 4
BOX_PIXELS = 84 // SIDE_BOXES
EXTRA_GOALS = 4


def box_start(x):
    return (x // BOX_PIXELS) * BOX_PIXELS


def create_goal(agent):
    goal = np.zeros(shape=(84, 84, 1))
    start_x, start_y = map(box_start, agent)
    goal[start_x:start_x + BOX_PIXELS, start_y:start_y + BOX_PIXELS, 0] = 255
    return goal


def one_hot_encode(action, num_actions):
    one_hot = np.zeros(num_actions)
    one_hot[action] = 1
    return one_hot


def find_agent(obs):
    # looking for gray color that is on the agent
    image = obs[:, :, -1]
    indices = np.flatnonzero(image == 110)
    if len(indices) == 0:
        return None
    index = indices[0]
    x = index % 84
    y = index // 84
    return x, y  # pixel coord


def goal_reward(obs, goal):
    obs = obs.transpose(1, 2, 0)
    agent = find_agent(obs)  # pixel coords
    goal_reached = False
    if agent is not None:
        goal_reached = goal[agent] > 0  # boolean
    return float(goal_reached)  # make 1 or 0


def final_goal(trajectory):
    for experience in reversed(trajectory):
        _, _, _, _, next_obs, _ = experience
        # print(next_obs.shape)
        agent = find_agent(next_obs)
        if agent:
            return create_goal(agent)  # return goal for final obs in trajectory like in HER paper
    return None


def future_goals(i, trajectory):
    goals = []
    if i + 1 >= len(trajectory):
        return None
    steps = np.random.randint(i + 1, len(trajectory), EXTRA_GOALS)  # k extra goals
    for step in steps:
        _, _, _, _, next_obs, _ = trajectory[step]
        next_obs = next_obs.transpose(1, 2, 0)
        agent = find_agent(next_obs)
        if agent:
            goals.append(create_goal(agent))  # near future goals of agent position
    return goals


def sample_goal():
    random_agent_position = np.random.randint(0, 84, 2)  # random pixel coord of agent
    return create_goal(random_agent_position)
