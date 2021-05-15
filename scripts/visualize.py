import argparse
import time
import numpy
import torch

import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=True,
                    help="select the action with highest probability (default: True)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = utils.make_env(args.env, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)

print("MODEL DIR", model_dir)

agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')

mycumulativereward = 0
mycumulativeperf = 0
mycumulativeperffull = 0
runsNum = 0
for episode in range(args.episodes):
    obs = env.reset()
    myreward = 0
    myperf = 0
    myperffull = 0
    while True:
        env.render('human')
        # time.sleep(args.pause)
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)

        myreward += reward
        myperf += info['performance']
        myperffull += info['performance_full']

        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            #print("final info", info)
            runsNum += 1
            mycumulativereward += myreward
            mycumulativeperf += myperf
            mycumulativeperffull += myperffull
            print("final myreward", myreward)
            print("final myperf", myperf)
            print("final myperffull", myperffull)
            print("cumulative runs", runsNum)
            print("final cumulative myreward", mycumulativereward, "(", mycumulativereward / runsNum,")")
            print("final cumulative myperf", mycumulativeperf, "(", mycumulativeperf / runsNum,")")
            print("final cumulative myperffull", mycumulativeperffull, "(", mycumulativeperffull / runsNum,")")
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif to ",args.gif+".gif", "... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
