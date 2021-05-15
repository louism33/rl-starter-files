import argparse
import random
import time
from collections import deque
from statistics import mean

import tensorboardX
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from gym_minigrid.wrappers import *

import utils
from scripts.myvisualizeAgents import visualiseAndSave, dqn

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)
parser = argparse.ArgumentParser()
parser.add_argument("--visualise", required=False, type=bool, default=False,
                    help="show the env or not")

parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")

parser.add_argument("--testenv", required=True,
                    help="name of the environment to train on (REQUIRED)")

parser.add_argument("--model", required=False,
                    help="name of the trained model")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")

parser.add_argument("--test", type=bool, default=False,
                    help="use test set")

parser.add_argument("--episodes", type=int, default=0,
                    help="number of episodes to visualize")
parser.add_argument("--conv", type=int, default=0,
                    help="if you need a conv layer")
parser.add_argument("--eps_dec", type=int, default=300000,
                    help="epsilon decay")

parser.add_argument("--frames", type=int, default=950000,
                    help="number of frames")

parser.add_argument("--save_interval", type=int, default=500,
                    help="after how many episodes to save")

parser.add_argument("--log_interval", type=int, default=10000,
                    help="after how many episodes to log")

parser.add_argument("--gamma", type=float, default=0.99,
                    help="gamma")

parser.add_argument("--lr", type=float, default=0.00001,
                    help="learning rate")

parser.add_argument("--seed", type=int, default=1,
                    help="seed")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")

parser.add_argument("--dir", required=False,
                    help="folder name")

parser.add_argument("--testeveryNlogs", type=int, default=10,
                    help="how often to run the test set")

parser.add_argument("--testepisodes", type=int, default=1,
                    help="how many different test permutations there are")
parser.add_argument("--trainingepisodes", type=int, default=2,
                    help="how many different training permutations there are")

args = parser.parse_args()
seed = args.seed

if seed:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


visualMode = (args.visualise and args.model)

environmentKeyTEST = 'MiniGrid-ScalableOversightTEST-v0'
if not visualMode:
    environmentKey = 'MiniGrid-ScalableOversightTRAINING-v0'
else:
    if args.test:
        environmentKey = 'MiniGrid-ScalableOversightTEST-v0'
    else:
        environmentKey = 'MiniGrid-ScalableOversightVISUALISE-v0'

env = gym.make(environmentKey)
env = utils.MyOneHotPartialObsWrapper(env)
env = ImgObsWrapper(env)  # Get rid of the 'mission' field

myTestEnv = gym.make(environmentKeyTEST)
myTestEnv = utils.MyOneHotPartialObsWrapper(myTestEnv)
myTestEnv = ImgObsWrapper(myTestEnv)  # Get rid of the 'mission' field

if seed:
    env.seed(seed)
    myTestEnv.seed(seed)

assert env.action_space.n == myTestEnv.action_space.n

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        if args.conv > 0:
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=1, stride=1),
                nn.ReLU(),
            )
        else:
            self.features = nn.Sequential(
            )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 400),
            nn.ReLU(),
            nn.Linear(400, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon, visualMode):
        if visualMode or random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data.item()
        else:
            action = random.randrange(env.action_space.n)
        return action


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if visualMode:
    print("LOADING AGENT")

    model_dir = utils.get_model_dir(args.model)

    print("MODEL DIR", model_dir)

    if hasattr(env, 'my_shape'):
        model = CnnDQN(env.my_shape, env.action_space.n)
    else:
        model = CnnDQN(env.observation_space.shape, env.action_space.n)

    loaded_dict = torch.load(model_dir + "/status.pt")
    model.load_state_dict(loaded_dict["model_state"])

    print("load state frames:", loaded_dict['num_frames'], "updates:", loaded_dict['update'])

    model.to(device)
    model.eval()

else:
    print("new AGENT")
    if hasattr(env, 'my_shape'):
        model = CnnDQN(env.my_shape, env.action_space.n)
    else:
        model = CnnDQN(env.observation_space.shape, env.action_space.n)

if USE_CUDA:
    print("USE CUDA")
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = args.eps_dec

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
    -1. * frame_idx / epsilon_decay)

max_frames = args.frames
save_interval = args.save_interval

log_interval = args.log_interval

batch_size = 32
gamma = args.gamma

import datetime

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
if not visualMode:
    conv = args.conv

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_DQN_F-{args.frames}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name, args.dir)

    print("MODEL NAME", model_name, "model dir", model_dir)

else:
    model_name = args.model

# Load loggers and Tensorboard writer
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
if not visualMode:
    tb_writer = tensorboardX.SummaryWriter(model_dir)

# https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb

if not visualMode:
    txt_logger.info("starting training\n")

state = env.reset()
start_time = time.time()
losses = []

all_rewards = []
all_perfs = []
all_perfs_full = []
all_dirts = []
all_phones = []
all_buttons = []
all_messes = []

episode_reward = 0
episode_perf = 0
episode_perf_full = 0
episode_dirts = 0
episode_phones = 0
episode_buttons = 0
episode_messes = 0

permutes = 0
allPermutes = 0
buttonValue = 0

dones = 0
previous_dones = 0
log_update = 0
duration = 0
saved = 0

header = []

if visualMode:
    env.render('human')

testDuration = 0
testData = [0] * 15

dataCopy = []
def do_logging(
        testTrainingPerformanceFull=0,
        testTrainingReward=0,
        testTrainingPerformance=0,
        testTrainingButtons=0,
        testTrainingDirt=0,
        testTrainingPhones=0,
        testTrainingMesses=0,

        testTestPerformanceFull=0,
        testTestReward=0,
        testTestPerformance=0,
        testTestButtons=0,
        testTestDirt=0,
        testTestPhones=0,
        testTestMesses=0,

        numberOfTestRuns=0,
        numSaves=0,

        final_log=False):

    global dataCopy

    if not final_log:
        timeSinceStart = time.time() - start_time
        fps = frame_idx / timeSinceStart
        duration = int(time.time() - start_time)
        data = [log_update, frame_idx, fps, duration, dones]
        newDones = (dones - previous_dones)
        txt_logger.info(("all_dirts length: ", len(all_dirts), "newDones", newDones))
        data += [mean(all_rewards)]
        all_rewards.clear()
        data += [mean(all_perfs)]
        all_perfs.clear()
        meanPerfFull = mean(all_perfs_full)
        data += [meanPerfFull]
        all_perfs_full.clear()
        data += [mean(all_dirts)]
        all_dirts.clear()
        data += [mean(all_phones)]
        all_phones.clear()
        data += [mean(all_buttons)]
        all_buttons.clear()
        data += [mean(all_messes)]
        all_messes.clear()
        data += [buttonValue]
        data += [allPermutes]
        data += [epsilon]

        # we save a copy of data, because the final log won't have these fields
        dataCopy = data.copy()
    else:
        data = dataCopy.copy()
        data[1] += 1 # increment frames by 1 for the final log

    # testing stuff
    # header += ["X_test_reward", "X_test_performance", "X_test_performance_full"]
    data += [testTestReward]
    data += [testTestPerformance]
    data += [testTestPerformanceFull]
    # more testing stuff
    # header += ["X_test_buttons", "X_test_dirts", "X_test_phones", "X_test_messes"]
    data += [testTestButtons]
    data += [testTestDirt]
    data += [testTestPhones]
    data += [testTestMesses]
    # train testing stuff
    # header += ["X_TRAINtest_reward", "X_TRAINtest_performance", "X_TRAINtest_performance_full"]
    data += [testTrainingReward]
    data += [testTrainingPerformance]
    data += [testTrainingPerformanceFull]
    # more train testing stuff
    # header += ["X_TRAINtest_buttons", "X_TRAINtest_dirts", "X_TRAINtest_phones", "X_TRAINtest_messes"]
    data += [testTrainingButtons]
    data += [testTrainingDirt]
    data += [testTrainingPhones]
    data += [testTrainingMesses]
    # header += ["X_all_performance_full"]
    data += [(testTestPerformanceFull + testTrainingPerformanceFull) / 2]

    # header += ["X_all_performance"]
    data += [(testTestPerformance + testTrainingPerformance)/2]

    # header += ["number_of_test_runs"]
    data += [numberOfTestRuns]
    # header += ["number_of_saves"]
    data += [numSaves]
    '''
                Note: U gives the update number, F the total number of frames, FPS the number of frames per second, D the 
                total duration, rR:μσmM the mean, std, min and max reshaped return per episode, 
                F:μσmM the mean, std, min and max number of frames per episode, H the entropy, V the value, pL the policy loss,
                vL the value loss and ∇ the gradient norm.
                performance:μσmM the mean, std, min and max performance per episode
                '''
    if final_log:
        txt_logger.info("FINAL LOG)")
    txt_logger.info(
        "\nL {} | U {} | FPS {:04.0f} | D {} | Dones {} "
        "| reward  {:.2f} "
        "| performance  {:.2f} "
        "| performanceFull  {:.2f} "
        "| dirt  {:.2f} "
        "| phones  {:.2f} "
        "| buttons  {:.2f} "
        "| messes  {:.2f} "
        "| buttonValue  {:.2f} "
        "| perm  {} "
        "| e  {:.2f} "
        "||| "
        "| TEST reward {:.2f} performance {:.2f} perf_full {:.2f} "
        "| TEST buttons {:.2f} phones {:.2f} dirt {:.2f} messes {:.2f}"
        "| TRAINING reward {:.2f} performance {:.2f} perf_full {:.2f} "
        "| TRAINING buttons {:.2f} phones {:.2f} dirt {:.2f} messes {:.2f}"

        "| allCombinedPerformanceFull  {:.2f} "
        "| numTestRuns  {:.2f} "
        "| numSaves  {:.2f} ".format(*data))
    csv_logger.writerow(data)
    csv_file.flush()
    for field, value in zip(header, data):
        tb_writer.add_scalar(field, value, frame_idx)
    pass

justTested = False
justTrainTested = False
numberOfTestRuns = 0

testTrainingReward = 0
testTrainingPerformance = 0
testTrainingPerformanceFull = 0
testTrainingButtons = 0
testTrainingDirt = 0
testTrainingPhones = 0
testTrainingMesses = 0

testTestReward = 0
testTestPerformance = 0
testTestPerformanceFull = 0
testTestButtons = 0
testTestDirt = 0
testTestPhones = 0
testTestMesses = 0
log_update = 0

if not visualMode:
    header = ["log_update", "frames", "FPS", "duration", "episodes"]
    header += ["X_return_mean"]
    header += ["X_performance_mean"]
    header += ["X_performance_full_mean"]
    header += ["X_dirt_mean"]
    header += ["X_phones_mean"]
    header += ["X_buttons_mean"]
    header += ["X_messes_mean"]
    header += ["buttonValue"]
    header += ["X_permutes"]
    header += ["X_epsilon"]
    #### test stuff
    header += ["X_test_reward", "X_test_performance", "X_test_performance_full"]
    header += ["X_test_buttons", "X_test_dirts", "X_test_phones", "X_test_messes"]
    header += ["X_TRAINtest_reward", "X_TRAINtest_performance", "X_TRAINtest_performance_full"]
    header += ["X_TRAINtest_buttons", "X_TRAINtest_dirts", "X_TRAINtest_phones", "X_TRAINtest_messes"]
    header += ["X_all_performance_full"]

    header += ["X_all_performance"]

    header += ["number_of_test_runs"]
    header += ["number_of_saves"]

    csv_logger.writerow(header)
    pass

for frame_idx in range(1, max_frames + 1):

    if visualMode:
        env.render('human')

    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon, visualMode)

    next_state, reward, done, info = env.step(action)

    allMyData = info.get('allMyData', None)
    if allMyData:
        txt_logger.info(allMyData)

    if not visualMode:
        replay_buffer.push(state, action, reward, next_state, done)

    state = next_state

    episode_reward += reward
    episode_perf += info['performance']
    episode_perf_full += info['performance_full']
    episode_dirts += info['dirt_cleaned']
    episode_phones += info['phones_cleaned']
    episode_buttons += info['button_presses']
    episode_messes += info['messes_cleaned']

    permutes += info['numberOfPermutes']

    buttonValue = info['buttonValue']

    if done:
        duration = int(time.time() - start_time)
        dones += 1

        if args.episodes > 0 and dones > args.episodes:
            break

        state = env.reset()

        if visualMode:
            txt_logger.info("final info", info)
            txt_logger.info("final myreward", episode_reward)

        all_rewards.append(episode_reward)
        all_perfs.append(episode_perf)
        all_perfs_full.append(episode_perf_full)
        all_dirts.append(episode_dirts)

        all_phones.append(episode_phones)
        all_buttons.append(episode_buttons)
        all_messes.append(episode_messes)

        episode_reward = 0
        episode_perf = 0
        episode_perf_full = 0
        episode_dirts = 0
        episode_phones = 0
        episode_buttons = 0
        episode_messes = 0
        allPermutes = permutes
        permutes = 0

        if not visualMode and (
                (save_interval > 0 and dones % save_interval == 0 and (justTrainTested and justTested))
                or
                (log_update == 0 and saved == 0)):
            status = {"num_frames": frame_idx, "update": dones,
                      "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict()
                      }
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved at frames " + str(frame_idx))
            txt_logger.info(("Time taken: ", str(datetime.timedelta(seconds=duration)), "name", str(model_name)))
            saved += 1
            justTested = False
            justTrainTested = False

    if not visualMode and frame_idx % log_interval == 0 and dones > 0:
        log_update += 1
        # time to test
        if log_update > 0 and log_update % args.testeveryNlogs == 0 and saved > 0 and not justTested and not justTrainTested:
            justTested = True
            justTrainTested = True
            txt_logger.info("***running test and training set***")
            numberOfTestRuns += 1
            testTrainingReward, testTrainingPerformance, testTrainingPerformanceFull, testTrainingButtons, testTrainingDirt, testTrainingPhones, testTrainingMesses = visualiseAndSave(
                args.env,
                model_name,
                args.seed,
                args.trainingepisodes * 4,
                txt_logger,
                gifName="training",
                save=False,
                dir=args.dir,
                agentType=dqn,
                CNNCLASS=CnnDQN)

            txt_logger.info(("testTrainingReward", testTrainingReward, "testTrainingPerformance",
                             testTrainingPerformance, "testTrainingPerformanceFull", testTrainingPerformanceFull))

            testTestReward, testTestPerformance, testTestPerformanceFull, testTestButtons, testTestDirt, testTestPhones, testTestMesses = visualiseAndSave(
                args.testenv, model_name,
                args.seed,
                args.testepisodes * 4,
                txt_logger,
                gifName="testing",
                save=False,
                dir=args.dir,
                agentType=dqn,
                CNNCLASS=CnnDQN)

            txt_logger.info(("testTestReward", testTestReward, "testTestPerformance", testTestPerformance,
                             "testTestPerformanceFull", testTestPerformanceFull))

        do_logging(
            testTrainingPerformanceFull=testTrainingPerformanceFull,
            testTrainingReward=testTrainingReward,
            testTrainingPerformance=testTrainingPerformance,
            testTrainingButtons=testTrainingButtons,
            testTrainingDirt=testTrainingDirt,
            testTrainingPhones=testTrainingPhones,
            testTrainingMesses=testTrainingMesses,
            testTestPerformanceFull=testTestPerformanceFull,
            testTestReward=testTestReward,
            testTestPerformance=testTestPerformance,
            testTestButtons=testTestButtons,
            testTestDirt=testTestDirt,
            testTestPhones=testTestPhones,
            testTestMesses=testTestMesses,
            numberOfTestRuns=numberOfTestRuns,
            numSaves=saved,
            final_log=False)

        previous_dones = dones
        pass

    if not visualMode and len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size)
        losses.append(loss.data.item)

### finished training, now time to see the absolute final performance of the agent on training and test set, save gifs and make a final log

txt_logger.info("\nAll done, quick save before testing")

txt_logger.info(("prefinal saving to", model_dir))
saved += 1
frame_idxplus1 = frame_idx
status = {"num_frames": frame_idxplus1, "update": dones,
          "model_state": model.state_dict(),
          "optimizer_state": optimizer.state_dict()
          }
utils.save_status(status, model_dir)
txt_logger.info("Status saved at frames " + str(frame_idxplus1))
txt_logger.info(("Time taken: ", str(datetime.timedelta(seconds=duration)), "name", str(model_name)))
saved += 1

txt_logger.info("\nAll saved, running last training for the final results to match up")

finalTrainingReward, finalTrainingPerformance, finalTrainingPerformanceFull, finalTrainingButtons, finalTrainingDirt, finalTrainingPhones, finalTrainingMesses = visualiseAndSave(
    args.env,
    model_name,
    args.seed,
    args.trainingepisodes * 4,
    txt_logger,
    gifName="training",
    save=True,
    dir=args.dir,
    agentType=dqn,
    CNNCLASS=CnnDQN)
txt_logger.info(("finalTrainingReward", finalTrainingReward, "finalTrainingPerformance", finalTrainingPerformance,
                 "finalTrainingPerformanceFull", finalTrainingPerformanceFull,
                 "finalTrainingButtons", finalTrainingButtons, "finalTrainingDirt", finalTrainingDirt,
                 "finalTrainingPhones", finalTrainingPhones, "finalTrainingMesses", finalTrainingMesses))
txt_logger.info("\nDone training, running last test for the final results to match up")

finalTestReward, finalTestPerformance, finalTestPerformanceFull, finalTestButtons, finalTestDirt, finalTestPhones, finalTestMesses = visualiseAndSave(
    args.testenv, model_name,
    args.seed,
    args.testepisodes * 4,
    txt_logger,
    gifName="testing",
    save=True,
    dir=args.dir,
    agentType=dqn,
    CNNCLASS=CnnDQN)
txt_logger.info(("finalTestReward", finalTestReward, "finalTestPerformance", finalTestPerformance,
                 "finalTestPerformanceFull", finalTestPerformanceFull,
                 "finalTestButtons", finalTestButtons, "finalTestDirt", finalTestDirt, "finalTestPhones",
                 finalTestPhones, "finalTestMesses", finalTestMesses))
txt_logger.info("\nAll done, proceeding to final log\n")

do_logging(
    testTrainingPerformanceFull=finalTrainingPerformanceFull,
    testTrainingReward=finalTrainingReward,
    testTrainingPerformance=finalTrainingPerformance,
    testTrainingButtons=finalTrainingButtons,
    testTrainingDirt=finalTrainingDirt,
    testTrainingPhones=finalTrainingPhones,
    testTrainingMesses=finalTrainingMesses,

    testTestPerformanceFull=finalTestPerformanceFull,
    testTestReward=finalTestReward,
    testTestPerformance=finalTestPerformance,
    testTestButtons=finalTestButtons,
    testTestDirt=finalTestDirt,
    testTestPhones=finalTestPhones,
    testTestMesses=finalTestMesses,

    numberOfTestRuns=numberOfTestRuns,
    numSaves=saved,
    final_log=True)

txt_logger.info(("FINAL saving to", model_dir))
saved += 1
status = {"num_frames": frame_idxplus1, "update": dones,
          "model_state": model.state_dict(),
          "optimizer_state": optimizer.state_dict()
          }
utils.save_status(status, model_dir)
txt_logger.info("Status saved at frames " + str(frame_idxplus1))
txt_logger.info(("Time taken: ", str(datetime.timedelta(seconds=duration)), "name", str(model_name)))

txt_logger.info(("Status saved, Time taken: ", str(datetime.timedelta(seconds=int(time.time() - start_time))), "name",
                 str(model_name)))

txt_logger.info(("MODEL NAME IS ", model_name))
txt_logger.info(("Time taken: ", str(datetime.timedelta(seconds=int(time.time() - start_time)))))

txt_logger.info("thank you for your patience")
