import argparse
import collections
import math
import os

import matplotlib.pyplot as plt
import pandas as pd

def my_plot_function(showGraph=False, x_axis="episodes", algo="ppo", env=''):
    masterDataFrameList = {}

    dfs = []
    divBy = 0
    avg = None

    for subdir, dirs, files in os.walk(os.path.join(storagePath, args.dir)):
        for sd in dirs:
            df_ = pd.read_csv(os.path.join(storagePath, args.dir, sd, "log.csv"), sep=",")
            dfs.append(df_)

            if avg is None:
                avg = df_.copy()

                for header in list(df_.copy()):
                    masterDataFrameList[header] = df_[ [header, x_axis]]

            else:
                avg += df_.copy()

                for header in list(df_.copy()):
                    masterDataFrameList[header][header+str(len(dfs))] = df_[header]

            divBy += 1

    avg = avg / divBy

    for index, whatToPlot in enumerate(separate_graphs):
        if len(whatToPlot) == 0:
            continue

        biggestLastY = -100000000
        smallestLastY = 100000000

        ax = None

        horizontalLine1 = None
        horizontalLine2 = None

        desiredTitle = "untitled"

        def sort_(x):
            if whatToPlot.get(x[0], {}).get("secondary_y", 0):
                return 1
            return 0
        # we order the dict, because if we graph a separate_y before a regular one, the std shading is broken :'(
        # so we try to graph separate_y last
        masterDataFrameList = collections.OrderedDict(sorted(masterDataFrameList.items(), key=lambda x:sort_(x)))

        for colName in masterDataFrameList:
            if colName == x_axis or colName not in whatToPlot or whatToPlot[colName].get('ignore', False):
                continue

            if whatToPlot[colName].get('desiredTitle', False):
                desiredTitle = whatToPlot[colName]['desiredTitle']

            if whatToPlot[colName].get('horizontalLine1', None) is not None:
                horizontalLine1 = whatToPlot[colName].get('horizontalLine1', False)

            if whatToPlot[colName].get('horizontalLine2', None) is not None:
                horizontalLine2 = whatToPlot[colName].get('horizontalLine2', False)

            thisCompoundDf = masterDataFrameList[colName]

            plotMe = thisCompoundDf.loc[:,[c for c in thisCompoundDf.columns if c != x_axis]].mean(axis=1).to_frame()
            plotMe.columns = [colName]
            std = thisCompoundDf.loc[:, [c for c in thisCompoundDf.columns if c != x_axis]].std(axis=1)

            if whatToPlot[colName]['normalise']:
                plotMe[colName] = (plotMe[colName] - plotMe[colName].min()) / (
                        plotMe[colName].max() - plotMe[colName].min())

            assert len(thisCompoundDf[colName]) == len(std)

            addedValues = plotMe[colName].add(std).values
            plotMe = plotMe.assign(stdPLUS=addedValues)
            plotMe = plotMe.assign(justSTD=std)
            subbedValues = plotMe[colName].subtract(std).values
            plotMe = plotMe.assign(stdMINUS=subbedValues)

            if x_axis == "episodes":
                plotMe = plotMe.assign(episodes=thisCompoundDf[x_axis].values)
            elif x_axis == "frames":
                plotMe = plotMe.assign(frames=thisCompoundDf[x_axis].values)

            annotateAllChanges = whatToPlot[colName]['annotateAllChanges'] if not ONLY_LAST_VALUE_OVERWRITE else False

            # https://stackoverflow.com/questions/8409095/set-markers-for-individual-points-on-a-line-in-matplotlib
            percentilesPretty = [colName]

            plot_kwargs = {
                'x': x_axis,
                'y': percentilesPretty,
                'figsize':(12, 8),
                'label': [whatToPlot.get(n, {}).get('alias', n) for n in percentilesPretty]
            }
            if ax is not None: plot_kwargs['ax'] = ax
            if whatToPlot[colName].get('secondary_y', False): plot_kwargs['secondary_y'] = True
            if whatToPlot[colName].get('desiredColour', False): plot_kwargs['color'] = whatToPlot[colName]['desiredColour']

            ax = plotMe.plot(**plot_kwargs)

            fill_between_kwargs = {
                'alpha': 0.2,
            }
            if whatToPlot[colName].get('desiredColour', False): fill_between_kwargs['color'] = whatToPlot[colName][
                'desiredColour']
            plt.fill_between(plotMe[x_axis], plotMe['stdPLUS'], plotMe['stdMINUS'], **fill_between_kwargs)

            prev = None
            for pi, colNamePretty in enumerate(percentilesPretty):

                for i, (a, b) in enumerate(zip(plotMe[x_axis], plotMe[colNamePretty])):

                    lastValue = i+1 == len(plotMe[x_axis])
                    if b != prev or lastValue:
                        plusSTD = round(plotMe['justSTD'][i], 2)

                        if whatToPlot[colName].get('hidePlusMinus', False):
                            writeMeOnGraph = str(round(b, 2))
                        else:
                            writeMeOnGraph = str(round(b, 2)) + " (+-" + str(plusSTD) + ")"

                        if annotateAllChanges or lastValue:
                            ### simple annotation
                            if not NO_NUMBERS and not whatToPlot[colName].get('hideValue', False):
                                plt.annotate(writeMeOnGraph, xy=(a, b))
                        if lastValue:
                            biggestLastY = max(biggestLastY, b)
                            smallestLastY = min(smallestLastY, b)
                        prev = b


        if horizontalLine1 is not None:
            plt.axhline(y=horizontalLine1, color='firebrick', linestyle='--', label="Random Full")
        if horizontalLine2 is not None:
            plt.axhline(y=horizontalLine2, color='midnightblue', linestyle=':', label="Random Button")

        plt.title(desiredTitle + ", " + algo + " " + env)

        ### axis modification
        axes = plt.gca()

        plt.grid(True)

        plt.savefig(os.path.join(storagePath, args.dir, "graph" + str(index) + "_" + desiredTitle + "_" + algo + "_" + env + ".pdf"))

        if showGraph:
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="folder name")
    parser.add_argument("--x_axis", default="frames",
                        help="x_axis")
    parser.add_argument("--algo", default="ppo",
                        help="which algo to save under")
    parser.add_argument("--env", default=False,
                        help="which env to save under")
    parser.add_argument("--storagePath", required=True, default=True,
                        help="storagePath")
    parser.add_argument("--showGraph", required=False, default=True, type=int,
                        help="plot.show() or not")

    args = parser.parse_args()

    storagePath = args.storagePath

    horizontal = -2
    horizontal3 = -1/2

    x_axis = args.x_axis
    x_axis = "frames"
    x_axis = "episodes"

    algo = args.algo
    algo = "ppo"
    # algo = "dqn"

    env = args.env if args.env else ''
    env = "(2, 2, 1, 2, 1)"

    ONLY_LAST_VALUE_OVERWRITE = True
    ONLY_LAST_VALUE_OVERWRITE = False
    NO_NUMBERS = True
    NO_NUMBERS = False

    realGraph0Perf = {
        "X_all_performance_full" : { #     data += [(finalTestPerformanceFull + finalTrainingPerformanceFull)/2]
            "desiredTitle": "performance graph",

            "alias": "performance_full",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': u'#1f77b4', # blue

            "horizontalLine1": horizontal,
            "horizontalLine2": 0,
            "horizontalLine3": horizontal3,
        },
        "X_all_performance" : { #     data += [(finalTestPerformance + finalTrainingPerformance)/2]
            "alias": "performance_button",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': u'#ff7f0e', # orange
        },
        "X_performance_full" : {
            "alias": "performance_button",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': u'#ff7f0e', # orange
        },
    }
    # colours not done
    realGraph1Training = {
        "X_TRAINtest_reward" : {
            "desiredTitle": "reward, entropy and buttons in training",

            "alias": "return",
            "normalise": False,
            "annotateAllChanges": False,
            "secondary_y": True,
        },
        "X_epsilon": {
            "desiredTitle": "reward, entropy and buttons in training",

            "alias": "epsilon",
            "normalise": False,
            "annotateAllChanges": False,
            "hideValue": True,
            "hidePlusMinus": True,
        },
        "entropy" : {
            "alias": "entropy",
            "normalise": False,
            "annotateAllChanges": False,
            "hideValue": True,
            "hidePlusMinus": True,
        },
        "buttons_mean" : { # ppo
            "alias": "training_buttons",
            "normalise": False,
            "annotateAllChanges": False
        },
        "X_buttons_mean" : { # dqn
            "alias": "training_buttons",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#9467bd',  # purple
        },
        "X_TRAINtest_buttons" : {
            "desiredTitle": "reward, entropy and buttons in training",
            "alias": "test_buttons",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#9467bd',  # purple
        },
    }
    # colours not done
    realGraph2TrainingPerfAndReward = {
        "X_TRAINtest_reward": {
            "desiredTitle": "reward and performances in training graph",

            "alias": "training_reward",
            "normalise": False,
            "annotateAllChanges": False,
            "secondary_y": True,
        },
        "X_TRAINtest_performance": {
            "desiredTitle": "reward and performances in training graph",

            "alias": "training_performance_button",
            "normalise": False,
            "annotateAllChanges": False,

            "ignore": False,
        },
        "X_TRAINtest_performance_full": {
            "alias": "training_performance_full",
            "normalise": False,
            "annotateAllChanges": False
        },

    }
    # colours not done
    realGraph3TestPerfAndReward = {
        "X_test_reward": {
            "desiredTitle": "reward and performances in test graph",

            "alias": "test_reward",
            "normalise": False,
            "annotateAllChanges": False,
            "secondary_y": True,
        },
        "X_test_performance": {
            "alias": "test_performance_button",
            "normalise": False,
            "annotateAllChanges": False
        },
        "X_test_performance_full": {
            "alias": "test_performance_full",
            "normalise": False,
            "annotateAllChanges": False
        },
    }

    realGraph4TrainingPerfVSTestBreakdown = {
        "X_test_performance_full": {
            "desiredTitle": "training and test performance breakdown",

            "alias": "test_performance_full",
            "normalise": False,
            "annotateAllChanges": False,
            # "secondary_y": True,

            "horizontalLine1": horizontal,
            "horizontalLine2": 0,

            'desiredColour': u'#1f77b4',  # blue
        },
        "X_test_performance": {
            "alias": "test_performance_button",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': u'#ff7f0e',  # orange
        },

        "X_TRAINtest_performance_full": {
            "alias": "training_performance_full",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#17becf',  # light blue
        },

        "X_TRAINtest_performance": {
            "alias": "training_performance_button",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#FFA500',  # orange
        },
    }
    # colours not done
    realGraph5Buttons = {
        "buttons_mean" : { # ppo
            "alias": "training_buttons",
            "normalise": False,
            "annotateAllChanges": False
        },
        "X_buttons_mean" : { # dqn
            "alias": "training_buttons",
            "normalise": False,
            "annotateAllChanges": False
        },
        "X_TRAINtest_buttons" : {
            "alias": "test_buttons",
            "normalise": False,
            "annotateAllChanges": False
        },
        "X_epsilon": {
            "desiredTitle": "graph of buttons",

            "alias": "epsilon",
            "normalise": False,
            "annotateAllChanges": False
        },
        "entropy": {
            "desiredTitle": "graph of buttons",

            "alias": "entropy",
            "normalise": False,
            "annotateAllChanges": False
        },
    }
    #colours not done
    realGraph6Phones = {

        "buttons_mean" : { # ppo
            "desiredTitle": "graph of phones",

            "alias": "training_buttons",
            "normalise": False,
            "annotateAllChanges": False
        },
        "X_buttons_mean" : { # dqn
            "alias": "training_buttons",
            "normalise": False,
            "annotateAllChanges": False
        },
        "X_TRAINtest_buttons" : {
            "desiredTitle": "graph of phones",
            "alias": "test_buttons",
            "normalise": False,
            "annotateAllChanges": False
        },

        "X_TRAINtest_phones": {
            "alias": "training_phones",
            "normalise": False,
            "annotateAllChanges": False
        },

        "X_test_phones" : {
            "alias": "test_phones",
            "normalise": False,
            "annotateAllChanges": False
        },
    }

    realGraph7PhonesButtonDirtTraining = {
        "X_TRAINtest_phones": {
            "desiredTitle": "graph of phones, dirts and buttons in training",
            "alias": "training_phones",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#d62728', # red
        },
        "X_TRAINtest_buttons": {
            "alias": "training_buttons",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#9467bd', #purple
        },

        "X_TRAINtest_dirts": {
            "alias": "training_dirts",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#2ca02c', #green
        },
        "X_TRAINtest_messes": {
            "alias": "training_messes",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#7f7f7f',  # grey
        },
    }

    realGraph8PhonesButtonDirtTest = {
        "X_test_phones": {
            "desiredTitle": "graph of phones, dirts and buttons on the test set",
            "alias": "test_phones",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#d62728',  # red
        },

        "X_test_buttons": {
            "alias": "test_buttons",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#9467bd',  # purple
        },
        "X_test_dirts": {
            "alias": "test_dirts",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#2ca02c',  # green
        },

        "X_test_messes": {
            "alias": "test_messes",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#7f7f7f',  # grey
        },

    }

    realGraph9PhonesButtonDirtAll = {
        "X_TRAINtest_phones": {
            "desiredTitle": "graph of phones, dirts and buttons in training and on the test set",
            "alias": "training_phones",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#ff7f0e', # orange (related to red)
        },
        "X_test_phones": {
            "alias": "test_phones",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#d62728',  # red
        },

        "X_TRAINtest_buttons": {
            "alias": "training_buttons",
            "normalise": False,
            "annotateAllChanges": False  ,

            'desiredColour': '#e377c2', # pink (related to purple)
        },
        "X_test_buttons": {
            "alias": "test_buttons",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#9467bd',  # purple
        },

        "X_TRAINtest_dirts": {
            "alias": "training_dirts",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#bcbd22', # light green/yellow (related to green)
        },
        "X_test_dirts": {
            "alias": "test_dirts",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#2ca02c',  # green
        },

        "X_TRAINtest_messes": {
            "alias": "training_messes",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#8c564b', # brown (related to grey)
        },
        "X_test_messes": {
            "alias": "test_messes",
            "normalise": False,
            "annotateAllChanges": False,

            'desiredColour': '#7f7f7f',  # grey
        },

    }


    separate_graphs = [
        realGraph0Perf,
        realGraph1Training,
        realGraph2TrainingPerfAndReward,
        realGraph3TestPerfAndReward,
        realGraph4TrainingPerfVSTestBreakdown,
        realGraph5Buttons,
        realGraph6Phones,
        realGraph7PhonesButtonDirtTraining,
        realGraph8PhonesButtonDirtTest,
        realGraph9PhonesButtonDirtAll,
    ]

    showGraph = False if args.showGraph == 0 else True

    my_plot_function(showGraph=showGraph, x_axis=x_axis, algo=algo, env=env)