import numpy
import torch
import torch.autograd as autograd
from gym_minigrid.wrappers import *

import utils

ppo = "ppo/q2c"
dqn = "dqn"

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

def visualiseAndSave(envStr, model_name, seed, numEpisodes, txt_logger, gifName="test", save = False, dir = None, agentType=ppo, CNNCLASS=None):

    if agentType != ppo and agentType != dqn:
        raise Exception

    utils.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = utils.make_env(envStr, seed)

    model_dir = utils.get_model_dir(model_name, dir)

    if agentType == ppo:
        agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                            device=device, argmax=True, use_memory=False, use_text=False)
    else:
        if hasattr(env, 'my_shape'):
            model = CNNCLASS(env.my_shape, env.action_space.n)
        else:
            model = CNNCLASS(env.observation_space['image'].shape, env.action_space.n)

        loaded_dict = torch.load(model_dir + "/status.pt")
        model.load_state_dict(loaded_dict["model_state"])

        print("For Test load state frames:", loaded_dict['num_frames'], "updates:", loaded_dict['update'])

        model.to(device)
        model.eval()
        if USE_CUDA:
            print("USE CUDA")
            model = model.cuda()

    if save:
        from array2gif import write_gif
        frames = []

    mycumulativereward = 0
    mycumulativeperf = 0
    mycumulativeperffull = 0

    mycumulativeButtons = 0
    mycumulativePhones = 0
    mycumulativeDirts = 0
    mycumulativeMesses = 0

    runsNum = 0
    for episode in range(numEpisodes):
        obs = env.reset()
        myreward = 0
        myperf = 0
        myperffull = 0

        myButtons = 0
        myPhones = 0
        myDirts = 0
        myMesses = 0

        while True:
            if save:
                frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

            if agentType == ppo:
                action = agent.get_action(obs)
            else:
                action = model.act(obs['image'], 0, True)  # epsilon == 0 so no exploration

            obs, reward, done, info = env.step(action)

            myreward += reward
            myperf += info['performance']
            myperffull += info['performance_full']

            myButtons += info['button_presses']
            myPhones += info['phones_cleaned']
            myDirts += info['dirt_cleaned']
            myMesses += info['messes_cleaned']

            if agentType == ppo:
                agent.analyze_feedback(reward, done)

            if done:
                runsNum += 1
                mycumulativereward += myreward
                mycumulativeperf += myperf
                mycumulativeperffull += myperffull

                mycumulativeButtons += myButtons
                mycumulativePhones += myPhones
                mycumulativeDirts += myDirts
                mycumulativeMesses += myMesses

                averageReward = mycumulativereward / runsNum
                averagePerformance = mycumulativeperf / runsNum
                averagePerformanceFull = mycumulativeperffull / runsNum

                averageButtons = mycumulativeButtons / runsNum
                averageDirts = mycumulativeDirts / runsNum
                averagePhones = mycumulativePhones / runsNum
                averageMesses = mycumulativeMesses / runsNum
                break


    if save:
        saveMeAs = model_dir + "/" + model_name + gifName + ".gif"
        txt_logger.info(("Saving gif to ", saveMeAs, "... "))
        write_gif(numpy.array(frames), saveMeAs, fps=1/0.3)
        txt_logger.info("Done.")

    return averageReward, averagePerformance, averagePerformanceFull, averageButtons, averageDirts, averagePhones, averageMesses

