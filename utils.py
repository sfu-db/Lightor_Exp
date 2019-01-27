from math import factorial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from highlight_initializer import *
import numpy as np
import copy

def plot_peak(dataset, f, selected_sliding_window, start_of_highlight):
    font_size = 33
    start = start_of_highlight - 20
    end = selected_sliding_window + 35
    time_list = []

    for c in dataset.chat[f]:
        if c[-1] >= start and c[-1] <= end:
            time_list.append(c[-1])
        elif c[-1] > end:
            break

    smoothed = []
    re = {}
    k = start
    bin_size = 3
    win_size = 7
    polyorder = 4
    while k <= end:
            re[k] = 0
            k += bin_size
    for t in time_list:
        for b in re:
            if t > b and t <= b + bin_size:
                re[b] += 1
                break
    re = sorted(re.items(), key=lambda x: x[0])
    x = [j[0] for j in re]
    y = np.asarray([j[1] for j in re])
    yhat = savitzky_golay(y, win_size, polyorder)
    p = get_peak(x, yhat)
    revised_x = []
    revised_y = []
    print(x)
    for i in range(len(x)):
        revised_x.append(x[i])
        revised_y.append(yhat[i])
    plt.figure(figsize=(20,10))
    ax = plt.subplot(111)

    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")

    plt.xlabel('Timestamp(s)', fontsize=font_size * 1.5)
    plt.ylabel('Message Number', fontsize=font_size * 1.5)

    plt.hist(time_list, bins=np.arange(min(revised_x[0], time_list[0]), max(revised_x[-1], time_list[-1]), 1), edgecolor='black')
    plt.xticks(np.arange(min(revised_x[0], time_list[0]), max(revised_x[-1], time_list[-1]), 10), fontsize=font_size)
    plt.yticks(np.arange(0, 20, 5), fontsize=font_size)
    plt.ylim(0, 15)
    plt.plot(revised_x, revised_y, '-b', label='smoothed curve')

    plt.plot([p[0]], [p[1]], 'r*', markersize=font_size, label='peak')
    plt.plot([start_of_highlight, start_of_highlight], [0, 10], '--g', linewidth=3, label='start of highlight')
    plt.legend(fontsize=font_size * 1.5)
    plt.show()

def show_features(dataset):
    matplotlib.rc('xtick', labelsize=25)
    matplotlib.rc('ytick', labelsize=25)

    f, a = plt.subplots(nrows=1, ncols=3, figsize=(40,10))
    labels = ['Message number', 'Message length', 'Message similarity']
    for j in range(3):
        pos = [i[j] for i in dataset.true_examples]
        neg = [i[j] for i in dataset.false_examples]
        if j == 2:
            pos = [1 - p for p in pos]
            neg = [1 - p for p in neg]
        min_ = min(min(pos), min(neg))
        max_ = max(max(pos), max(neg))
        binwidth = (max_ - min_) / 20
        bins = np.arange(min_, max_ + binwidth, binwidth)
        r = a[j].hist([pos, neg], bins=bins, stacked=True, color=['#C11E2F', '#A5C9F4'], label=['highlight', 'non highlight'], rwidth=0.8)
        a[j].set_ylim(0, 20)
        a[j].set_yticks(np.arange(0, 25, 5))
        a[j].set_xlabel(labels[j], fontsize=40)
        a[j].set_ylabel('Frequency', fontsize=40)
        a[j].legend(fontsize=40)
    plt.show()

def plot_precision(prec_list, xLabel, yLabel, k, yScale, yTick):
    f = plt.figure(figsize=(7,3.5))

    plt.xlabel(xLabel, fontsize=21)
    plt.ylabel(yLabel, fontsize=21)


    ax = plt.subplot(111)

    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    ax.grid(axis = 'y', linestyle='dotted')

    plt.ylim(yScale[0], yScale[1])
    x_list = list(range(1, k + 1))
    plt.yticks(yTick, fontsize=18)
    plt.xticks(np.arange(min(x_list), max(x_list)+1, 1), fontsize=18)

    for prec in prec_list:
        plt.plot(x_list, prec[0], prec[1], label=prec[2], markersize=10)
    plt.legend(fontsize=15,loc=3, framealpha=0.3)
    plt.show()
