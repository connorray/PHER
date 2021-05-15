import torch
import os
import random
from argparse import ArgumentParser
from atari_wrappers import *
from dqn import VanillaDQNCUDA
from memory import ReplayBuffer
from utils import *
import pickle


def optimize(model, target_model, batch, num_actions, criterion, optimizer):
    observations, actions, rewards, next_observations, dones = batch
    next_state_action_values = target_model(next_observations)
    next_state_action_values[dones] = 0.0
    state_action_values = torch.from_numpy(rewards).to("cuda") + DISCOUNT_FACTOR_GAMMA * torch.max(
        next_state_action_values, dim=1).values
    one_hot_actions = np.array([one_hot_encode(action, num_actions) for action in actions])
    expected_state_action_values = state_action_values * (
            1 - torch.from_numpy(dones.astype(float)).to("cuda")) - torch.from_numpy(dones.astype(float)).to("cuda")
    state_action_values = model(observations)
    state_action_values = torch.sum(torch.multiply(state_action_values, torch.from_numpy(one_hot_actions).to("cuda")),
                                    dim=1)
    loss = criterion(expected_state_action_values, state_action_values)
    optimizer.zero_grad()
    loss.backward()
    # clip norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # grad clipping
    optimizer.step()


def update_epsilon(step):
    return max(EPSILON_FINAL, (EPSILON_FINAL - EPSILON_START) / EPSILON_STEPS * step + EPSILON_START)


def agent_act(env, model, observation, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(
            model(np.expand_dims(observation, axis=0)).cpu().detach().numpy())
    return action


def evaluate(env, model, view=False, eval_steps=EVAL_STEPS):
    model.eval()
    done = True
    episode = 0
    total_ep_return = 0.0
    for step in range(1, eval_steps):
        if done:
            if episode > 0:
                print("Evaluation Episode {} Steps {} Episode Return {}".format(
                    episode,
                    episode_steps,
                    episode_return,
                ))
                total_ep_return += episode_return
            obs = np.array(env.reset())
            obs = obs.transpose((2, 0, 1))
            episode += 1
            episode_return = 0.0
            episode_steps = 0
            if view:
                env.render()
        else:
            obs = next_obs
        action = agent_act(env, model, obs, EPSILON_FINAL)
        next_obs, reward, done, _ = env.step(action)
        next_obs = np.array(next_obs)
        next_obs = next_obs.transpose((2, 0, 1))
        episode_return += reward
        episode_steps += 1
        if view:
            env.render()
    episode_return_avg = total_ep_return / episode
    model.train()
    return episode_return_avg


def save_model(model, step, savedir):
    filename = './{}/'.format(savedir)
    try:
        os.makedirs(filename)
    except FileExistsError:
        # directory already exists
        pass
    torch.save(model.state_dict(), filename + "{}.pt".format(step))
    print('Saved {}'.format(filename))


def train(env, env_eval, model, total_steps, view, criterion, optimizer, savedir, param_save_dir):
    try:
        os.mkdir("./params")
        print("Directory params Created")
    except FileExistsError:
        print("Directory params already exists")
    model_dir = "./params/{}".format(param_save_dir)
    try:
        os.mkdir(model_dir)
        print("Directory ", model_dir, " Created")
    except FileExistsError:
        print("Directory ", model_dir, " already exists")

    target_model = VanillaDQNCUDA(n_actions=env.action_space.n).to("cuda")
    memory = ReplayBuffer(MEM_SIZE)
    done = True
    episode = 0
    log_steps = 0
    rewards_history = []
    for step in range(1, total_steps + 1):
        try:
            if step % SAVE_FREQ == 0:
                save_model(model, step, savedir)
            if done:
                if episode > 0:
                    for i, experience in enumerate(trajectory):
                        obs, action, reward, next_obs, done = experience
                        memory.add(obs, action, reward, next_obs, done)
                    if log_steps >= LOG_EVERY:
                        log_steps = 0
                        episode_steps = step - episode_start_step
                        print(
                            "Episode: {} | Steps: {}/{} | Return: {}".format(
                                episode,
                                episode_steps,
                                step,
                                episode_return
                            ))
                trajectory = []
                episode_start_step = step
                obs = np.array(env.reset())
                obs = obs.transpose((2, 0, 1))
                episode += 1
                episode_return = 0.0
                epsilon = update_epsilon(step)
            else:
                obs = next_obs
            action = agent_act(env, model, obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            next_obs = np.array(next_obs)
            next_obs = next_obs.transpose((2, 0, 1))
            episode_return += reward
            trajectory.append((obs, action, reward, next_obs, done))

            if step >= EXPLORE_STEPS and step % UPDATE_EVERY == 0:
                if step % TARGET_UPDATE_EVERY == 0:
                    target_model.load_state_dict(model.state_dict())
                batch = memory.sample(BATCH_SIZE)
                optimize(model, target_model, batch, num_actions=env.action_space.n, criterion=criterion,
                         optimizer=optimizer)
            if step >= EXPLORE_STEPS and step % EVAL_EVERY == 0:
                episode_return_avg = evaluate(env_eval, model, view=view)
                print(
                    "Episode: {} | Steps: {} | Evaluation Return Avg: {}".format(
                        episode,
                        step,
                        episode_return_avg,
                    ))
                rewards_history.append(episode_return_avg)
            log_steps += 1
        except KeyboardInterrupt:
            del trajectory[:]
            del rewards_history[:]
            break
    pickle.dump([rewards_history],
                open(model_dir + '/' + "model_test_rewards.p", "wb+"))
    env.close()
    env_eval.close()
    torch.cuda.empty_cache()
    return rewards_history


def set_seed(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


def main(args):
    env = make_atari('{}NoFrameskip-v4'.format(args.env))
    set_seed(env, args.seed)
    savedir = 'models/DQN_cuda/{}/{}'.format(args.env, args.seed)
    param_save_dir = 'DQN_cuda_{}_{}'.format(args.env, args.seed)

    # Models
    num_actions = env.action_space.n
    model = VanillaDQNCUDA(n_actions=num_actions).to("cuda")

    # Optimizer and Criterion
    optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
    criterion = torch.nn.L1Loss()  # MAE loss

    env_train = wrap_deepmind(env, frame_stack=True, episode_life=True, clip_rewards=True)
    env_eval = wrap_deepmind(env, frame_stack=True)
    if args.eval:
        step = 5000000
        model.load_state_dict(torch.load("./{}/{}.pt".format(savedir, step)))
        evaluate(env_eval, model, view=args.view)
    else:
        _ = train(env_train, env_eval, model, MAX_STEPS,
                  view=args.view, criterion=criterion,
                  optimizer=optimizer,
                  savedir=savedir, param_save_dir=param_save_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', action='store', default='MontezumaRevenge')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--seed', action='store', type=int, default=1219)
    parser.add_argument('--view', action='store_true', default=False)
    main(parser.parse_args())
