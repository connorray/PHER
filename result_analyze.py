import matplotlib.pyplot as plt
import pickle
import numpy as np


def smooth_training_data(training_data, smooth_half_window):
    """
    Smooth training data using average of smooth window
    :param training_data: list of training data
    :param smooth_half_window: half window size
    :return: smooth_data
    """
    data_num = training_data.shape[0]
    smooth_data = np.zeros(data_num)
    for num in range(data_num):
        if num < smooth_half_window:
            smooth_data[num] = np.sum(training_data[:num + smooth_half_window + 1]) / (num + smooth_half_window + 1)
        elif num >= data_num - smooth_half_window:
            smooth_data[num] = np.sum(training_data[num - smooth_half_window:]) / (data_num - num + smooth_half_window)
        else:
            smooth_data[num] = np.sum(training_data[num - smooth_half_window:num + smooth_half_window + 1]) / (
                        2 * smooth_half_window + 1)
    return smooth_data


def read_multiple_models(model_dir, seeds, steps):
    test_reward_list = np.zeros((steps, len(seeds)))
    for num, seed in enumerate(seeds):
        [test_reward] = pickle.load(open(model_dir + str(seed) + '/model_test_rewards.p', 'rb'))
        smooth_test_reward = smooth_training_data(np.array(test_reward), 8)
        test_reward_list[:, num] = smooth_test_reward
    return test_reward_list.mean(axis=1), test_reward_list.std(axis=1)


def plot_multiple_mean_rewards(model_dir_list, label_list, color_list, seeds, steps, env_name):
    plt.figure()
    for i, model_dir in enumerate(model_dir_list, 0):
        label = label_list[i]
        color = color_list[i]
        reward_mean, reward_std = read_multiple_models(model_dir, seeds, steps)
        plt.plot([num for num in range(steps)], reward_mean, color, label=label)
        plt.fill_between([num for num in range(steps)], reward_mean - reward_std, reward_mean + reward_std,
                         alpha=0.2, color=color)
    plt.xlim([0, steps])
    plt.xlabel("Frames (x100k)")
    plt.ylabel("Average Rewards")
    plt.title(env_name + " (Avg over " + str(len(seeds)) + " models)")
    plt.legend()


if __name__ == '__main__':
    dir_list = ["./params/PHER_cuda_MontezumaRevenge_",
                "./params/DQN_cuda_MontezumaRevenge_"]

    label_list = ["PHER", "vanilla DQN"]
    color_list = ["b", "r"]
    seeds = [23, 1219, 222, 424, 605]
    plot_multiple_mean_rewards(dir_list, label_list, color_list, seeds, 50, "MontezumaRevenge")
    plt.show()
