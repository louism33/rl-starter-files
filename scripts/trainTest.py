import argparse
import datetime
import sys
import time

import tensorboardX
import torch
import torch_ac

import utils
from model import ACModel
from scripts.myvisualizeAgents import visualiseAndSave

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16, #todo changed proc to 1
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10 ** 7,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,  # basically exploration
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")

    parser.add_argument("--dir", required=False,
                        help="folder name")

    parser.add_argument("--testenv", required=False,
                        help="name of the environment to train on (REQUIRED)")

    parser.add_argument("--testeveryNlogs", type=int, default=10,
                        help="how often to run the test set")

    parser.add_argument("--testepisodes", type=int, default=2,
                        help="how many different test permutations there are")
    parser.add_argument("--trainingepisodes", type=int, default=2,
                        help="how many different training permutations there are")

    args = parser.parse_args()

    args.mem = args.recurrence > 1

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_F-{args.frames}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name, args.dir)
    print("SAVING MODEL AS", model_name, "in", model_dir)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    txt_logger.info(("SAVING MODEL AS ", str(model_name)))


    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments
    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed))
    txt_logger.info("Environments loaded\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    duration = 0
    log_update = 0

    saved_num = 0

    episodes = 0

    testTrainingReward = 0
    testTrainingPerformance = 0
    testTrainingPerformanceFull = 0

    testTrainingButtons = 0
    testTrainingPhones = 0
    testTrainingDirt = 0
    testTrainingMesses = 0

    testTestReward = 0
    testTestPerformance = 0
    testTestPerformanceFull = 0

    testTestButtons = 0
    testTestPhones = 0
    testTestDirt = 0
    testTestMesses = 0

    numberOfTestRuns = 0

    txt_logger.info("locals")
    txt_logger.info(locals())
    txt_logger.info("globals")
    txt_logger.info(globals())

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()

        if 'allMyData' in logs1:
            txt_logger.info(logs1['allMyData'])

        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        justTested = False
        justTrainTested = False

        # Print logs
        if update % args.log_interval == 0:

            if log_update > 0 and log_update % args.testeveryNlogs == 0 and saved_num > 0:
                justTested = True
                justTrainTested = True
                txt_logger.info("running test and training set")
                numberOfTestRuns += 1
                testTrainingReward, testTrainingPerformance, testTrainingPerformanceFull, testTrainingButtons, testTrainingDirt, testTrainingPhones, testTrainingMesses = visualiseAndSave(args.env,
                                                                                                               model_name,
                                                                                                               args.seed,
                                                                                                               args.trainingepisodes * 4,
                                                                                                               txt_logger,
                                                                                                               gifName="training",
                                                                                                               save=False,
                                                                                                               dir=args.dir)

                txt_logger.info(("testTrainingReward", testTrainingReward, "testTrainingPerformance",
                                 testTrainingPerformance, "testTrainingPerformanceFull", testTrainingPerformanceFull))

                testTestReward, testTestPerformance, testTestPerformanceFull, testTestButtons, testTestDirt, testTestPhones, testTestMesses = visualiseAndSave(args.testenv, model_name,
                                                                                                   args.seed,
                                                                                                   args.testepisodes * 4,
                                                                                                   txt_logger,
                                                                                                   gifName="testing",
                                                                                                   save=False,
                                                                                                   dir=args.dir)

                txt_logger.info(("testTestReward", testTestReward, "testTestPerformance", testTestPerformance,
                                 "testTestPerformanceFull", testTestPerformanceFull))


            log_update += 1
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])

            performance_per_episode = utils.synthesize(logs["performance_per_episode"])
            rperformance_per_episode = utils.synthesize(logs["reshaped_performance_per_episode"])

            buttons_per_episode = utils.synthesize(logs["buttons_per_episode"])
            reshaped_buttons_per_episode = utils.synthesize(logs["reshaped_buttons_per_episode"])

            phones_per_episode = utils.synthesize(logs["phones_per_episode"])
            reshaped_phones_per_episode = utils.synthesize(logs["reshaped_phones_per_episode"])

            dirt_per_episode = utils.synthesize(logs["dirt_per_episode"])
            reshaped_dirt_per_episode = utils.synthesize(logs["reshaped_dirt_per_episode"])

            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
            messes_per_episode = utils.synthesize(logs["messes_per_episode"])
            performance_full_per_episode = utils.synthesize(logs["performance_full_per_episode"])

            episodes = episodes + logs["episodesDone"]

            header = ["log_update", "update", "frames", "FPS", "duration", "episodes"]
            data = [log_update, update, num_frames, fps, duration, episodes]

            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            header += ["permutes"]
            data += [logs["numberOfPermutes"]]

            header += ["buttonValue"]
            data += [logs["buttonValue"]]

            header += ["performance_" + key for key in
                       performance_per_episode.keys()]
            data += performance_per_episode.values()

            header += ["performance_full_" + key for key in
                       performance_full_per_episode.keys()]
            data += performance_full_per_episode.values()

            header += ["buttons_" + key for key in buttons_per_episode.keys()]
            data += buttons_per_episode.values()
            header += ["messes_" + key for key in messes_per_episode.keys()]
            data += messes_per_episode.values()
            header += ["phones_" + key for key in phones_per_episode.keys()]
            data += phones_per_episode.values()

            header += ["dirt_" + key for key in dirt_per_episode.keys()]
            data += dirt_per_episode.values()

            # testing stuff
            header += ["X_test_reward", "X_test_performance", "X_test_performance_full"]
            data += [testTestReward]
            data += [testTestPerformance]
            data += [testTestPerformanceFull]

            # more testing stuff
            header += ["X_test_buttons", "X_test_dirts", "X_test_phones", "X_test_messes"]
            data += [testTestButtons]
            data += [testTestDirt]
            data += [testTestPhones]
            data += [testTestMesses]

            # train testing stuff
            header += ["X_TRAINtest_reward", "X_TRAINtest_performance", "X_TRAINtest_performance_full"]
            data += [testTrainingReward]
            data += [testTrainingPerformance]
            data += [testTrainingPerformanceFull]

            # more train testing stuff
            header += ["X_TRAINtest_buttons", "X_TRAINtest_dirts", "X_TRAINtest_phones", "X_TRAINtest_messes"]
            data += [testTrainingButtons]
            data += [testTrainingDirt]
            data += [testTrainingPhones]
            data += [testTrainingMesses]

            header += ["X_all_performance"]
            data += [(testTestPerformance + testTrainingPerformance)/2]

            header += ["X_all_performance_full"]
            data += [(testTestPerformanceFull + testTrainingPerformanceFull)/2]

            header += ["number_of_test_runs"]
            data += [numberOfTestRuns]
            header += ["number_of_saves"]
            data += [saved_num]

            '''
            Note: U gives the update number, F the total number of frames, FPS the number of frames per second, D the 
            total duration, E the episodes, rR:μσmM the mean, std, min and max reshaped return per episode, 
            F:μσmM the mean, std, min and max number of frames per episode, H the entropy, V the value, pL the policy loss,
            vL the value loss and ∇ the gradient norm.
            performance:μσmM the mean, std, min and max performance per episode
            '''
            txt_logger.info(
                    "\nL {} | U {} | F {:06} | FPS {:04.0f} | D {} | E {}"
                    # "| F:uomM {:.1f} {:.1f} {} {} "
                    "| H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | GradNorm {:.3f} "
                    "| return:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
                    "| perm  {:.2f} "
                    "| buttonValue  {:.2f} "
        
                    "| performance:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
                    "| perf_full:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
        
                    "| buttons:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
                    "| messes:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
                    "| phones:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
                    "| dirts:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
                    "| TEST reward {:.2f} performance {:.2f} perf_full {:.2f} "
                    "| TEST buttons {:.2f} phones {:.2f} dirt {:.2f} messes {:.2f}"
                    "| TRAINING reward {:.2f} performance {:.2f} perf_full {:.2f} "
                    "| TRAINING buttons {:.2f} phones {:.2f} dirt {:.2f} messes {:.2f}"
                    
                    "| allCombinedPerformance  {:.2f} "
                    "| allCombinedPerformanceFull  {:.2f} "
                    "| numTestRuns  {:.2f} "
                    "| numSaves  {:.2f} "

                    .format(*data))

            header += ["X_entropy"]
            data += [logs["X_entropy"]]

            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()

            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()

            header += ["r_performance_" + key for key in rperformance_per_episode.keys()]
            data += rperformance_per_episode.values()

            header += ["r_buttons_" + key for key in reshaped_buttons_per_episode.keys()]
            data += reshaped_buttons_per_episode.values()

            header += ["r_phones_" + key for key in reshaped_phones_per_episode.keys()]
            data += reshaped_phones_per_episode.values()

            header += ["r_dirt_" + key for key in reshaped_dirt_per_episode.keys()]
            data += reshaped_dirt_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status
        # we are experimenting with saving only after each test / traintest run
        # if args.save_interval > 0 and update % args.save_interval == 0:
        if args.save_interval > 0 and ((justTrainTested and justTested) or (log_update == 1 and saved_num == 0)): # and update % args.save_interval == 0:
            print("saving to", model_dir)
            saved_num += 1
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info(("Status saved, Time taken: ", str(datetime.timedelta(seconds=int(time.time() - start_time))), "name", str(model_name)))

    ### finished training, now time to see the absolute final performance of the agent on training and test set, save gifs and make a final log
    txt_logger.info("\nAll done, quick save before testing")

    print("prefinal saving to", model_dir)
    saved_num += 1
    status = {"num_frames": num_frames, "update": update,
              "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
    if hasattr(preprocess_obss, "vocab"):
        status["vocab"] = preprocess_obss.vocab.vocab
    utils.save_status(status, model_dir)
    # txt_logger.info("Status saved, name", str(model_name))
    txt_logger.info(("Status saved, Time taken: ", str(datetime.timedelta(seconds=int(time.time() - start_time))), "name", str(model_name)))

    txt_logger.info("\nAll saved, running last training for the final results to match up")

    finalTrainingReward, finalTrainingPerformance, finalTrainingPerformanceFull, finalTrainingButtons, finalTrainingDirt, finalTrainingPhones, finalTrainingMesses = visualiseAndSave(
        args.env,
        model_name,
        args.seed,
        args.trainingepisodes * 4,
        txt_logger,
        gifName="training",
        save=True,
        dir=args.dir)
    txt_logger.info(("finalTrainingReward", finalTrainingReward, "finalTrainingPerformance", finalTrainingPerformance, "finalTrainingPerformanceFull", finalTrainingPerformanceFull,
                     "finalTrainingButtons", finalTrainingButtons, "finalTrainingDirt", finalTrainingDirt, "finalTrainingPhones",finalTrainingPhones, "finalTrainingMesses", finalTrainingMesses))
    txt_logger.info("\nDone training, running last test for the final results to match up")

    finalTestReward, finalTestPerformance, finalTestPerformanceFull, finalTestButtons, finalTestDirt, finalTestPhones, finalTestMesses = visualiseAndSave(
        args.testenv, model_name,
        args.seed,
        args.testepisodes * 4,
        txt_logger,
        gifName="testing",
        save=True,
        dir=args.dir)
    txt_logger.info(("finalTestReward", finalTestReward, "finalTestPerformance", finalTestPerformance, "finalTestPerformanceFull", finalTestPerformanceFull,
                     "finalTestButtons", finalTestButtons, "finalTestDirt", finalTestDirt, "finalTestPhones",finalTestPhones, "finalTestMesses", finalTestMesses))
    txt_logger.info("\nAll done, proceeding to final log\n")

    log_update += 1
    fps = logs["num_frames"] / (update_end_time - update_start_time)
    duration = int(time.time() - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])

    performance_per_episode = utils.synthesize(logs["performance_per_episode"])
    rperformance_per_episode = utils.synthesize(logs["reshaped_performance_per_episode"])

    buttons_per_episode = utils.synthesize(logs["buttons_per_episode"])
    reshaped_buttons_per_episode = utils.synthesize(logs["reshaped_buttons_per_episode"])

    phones_per_episode = utils.synthesize(logs["phones_per_episode"])
    reshaped_phones_per_episode = utils.synthesize(logs["reshaped_phones_per_episode"])

    dirt_per_episode = utils.synthesize(logs["dirt_per_episode"])
    reshaped_dirt_per_episode = utils.synthesize(logs["reshaped_dirt_per_episode"])

    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
    messes_per_episode = utils.synthesize(logs["messes_per_episode"])
    performance_full_per_episode = utils.synthesize(logs["performance_full_per_episode"])

    episodes = episodes + logs["episodesDone"]

    header = ["log_update", "update", "frames", "FPS", "duration", "episodes"]
    data = [log_update, update, num_frames, fps, duration, episodes]

    header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

    header += ["return_" + key for key in return_per_episode.keys()]
    data += return_per_episode.values()

    header += ["permutes"]
    data += [logs["numberOfPermutes"]]

    header += ["buttonValue"]
    data += [logs["buttonValue"]]

    header += ["performance_" + key for key in
               performance_per_episode.keys()]
    data += performance_per_episode.values()

    header += ["performance_full_" + key for key in
               performance_full_per_episode.keys()]
    data += performance_full_per_episode.values()

    header += ["buttons_" + key for key in buttons_per_episode.keys()]
    data += buttons_per_episode.values()
    header += ["messes_" + key for key in messes_per_episode.keys()]
    data += messes_per_episode.values()
    header += ["phones_" + key for key in phones_per_episode.keys()]
    data += phones_per_episode.values()

    header += ["dirt_" + key for key in dirt_per_episode.keys()]
    data += dirt_per_episode.values()

    # testing stuff
    header += ["X_test_reward", "X_test_performance", "X_test_performance_full"]
    data += [finalTestReward]
    data += [finalTestPerformance]
    data += [finalTestPerformanceFull]

    # more testing stuff
    header += ["X_test_buttons", "X_test_dirts", "X_test_phones", "X_test_messes"]
    data += [finalTestButtons]
    data += [finalTestDirt]
    data += [finalTestPhones]
    data += [finalTestMesses]

    # train testing stuff
    header += ["X_TRAINtest_reward", "X_TRAINtest_performance", "X_TRAINtest_performance_full"]
    data += [finalTrainingReward]
    data += [finalTrainingPerformance]
    data += [finalTrainingPerformanceFull]

    # more train testing stuff
    header += ["X_TRAINtest_buttons", "X_TRAINtest_dirts", "X_TRAINtest_phones", "X_TRAINtest_messes"]
    data += [finalTrainingButtons]
    data += [finalTrainingDirt]
    data += [finalTrainingPhones]
    data += [finalTrainingMesses]

    header += ["X_all_performance"]
    data += [(finalTestPerformance + finalTrainingPerformance)/2]

    header += ["X_all_performance_full"]
    data += [(finalTestPerformanceFull + finalTrainingPerformanceFull)/2]

    header += ["number_of_test_runs"]
    data += [numberOfTestRuns]
    header += ["number_of_saves"]
    data += [saved_num]

    '''
    Note: U gives the update number, F the total number of frames, FPS the number of frames per second, D the 
    total duration, rR:μσmM the mean, std, min and max reshaped return per episode, 
    F:μσmM the mean, std, min and max number of frames per episode, H the entropy, V the value, pL the policy loss,
    vL the value loss and ∇ the gradient norm.
    performance:μσmM the mean, std, min and max performance per episode
    '''

    txt_logger.info(
        "\nABSOLUTE FINAL LOG! L {} | U {} | F {:06} | FPS {:04.0f} | D {} "
        "| H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | GradNorm {:.3f} "
        "| return:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
        "| perm  {:.2f} "
        "| buttonValue  {:.2f} "

        "| performance:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
        "| perf_full:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "

        "| buttons:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
        "| messes:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
        "| phones:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
        "| dirts:uomM  {:.2f} {:.2f} {:.2f} {:.2f} "
        "| TEST reward {:.2f} performance {:.2f} perf_full {:.2f} "
        "| TEST buttons {:.2f} phones {:.2f} dirt {:.2f} messes {:.2f}"
        "| TRAINING reward {:.2f} performance {:.2f} perf_full {:.2f} "
        "| TRAINING buttons {:.2f} phones {:.2f} dirt {:.2f} messes {:.2f}"

        "| allCombinedPerformance  {:.2f} "
        "| allCombinedPerformanceFull  {:.2f} "
        "| numTestRuns  {:.2f} "
        "| numSaves  {:.2f} "

            .format(*data))

    header += ["X_entropy"]
    data += [logs["X_entropy"]]

    header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
    data += num_frames_per_episode.values()

    header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
    data += rreturn_per_episode.values()

    header += ["r_performance_" + key for key in rperformance_per_episode.keys()]
    data += rperformance_per_episode.values()

    header += ["r_buttons_" + key for key in reshaped_buttons_per_episode.keys()]
    data += reshaped_buttons_per_episode.values()

    header += ["r_phones_" + key for key in reshaped_phones_per_episode.keys()]
    data += reshaped_phones_per_episode.values()

    header += ["r_dirt_" + key for key in reshaped_dirt_per_episode.keys()]
    data += reshaped_dirt_per_episode.values()

    if status["num_frames"] == 0:
        csv_logger.writerow(header)
    csv_logger.writerow(data)
    csv_file.flush()

    for field, value in zip(header, data):
        tb_writer.add_scalar(field, value, num_frames)

    print("FINAL saving to", model_dir)
    saved_num += 1
    status = {"num_frames": num_frames, "update": update,
              "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
    if hasattr(preprocess_obss, "vocab"):
        status["vocab"] = preprocess_obss.vocab.vocab
    utils.save_status(status, model_dir)
    txt_logger.info(("Status saved, Time taken: ", str(datetime.timedelta(seconds=int(time.time() - start_time))), "name", str(model_name)))

    txt_logger.info(("MODEL NAME IS ", model_name))
    txt_logger.info(("Time taken: ", str(datetime.timedelta(seconds=int(time.time() - start_time)))))

    print("thank you for your patience")