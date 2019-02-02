from math import factorial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import highlight_initializer
import numpy as np
import copy
from datetime import timezone, datetime, timedelta
from scipy import stats
import json
import seaborn as sns

def time(i):
    return "%03d-%02d"%(i / 60, i % 60)

def sec(i):
    t = i.split('-')
    return int(t[0]) * 60 + int(t[1])

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
    yhat = highlight_initializer.savitzky_golay(y, win_size, polyorder)
    p = highlight_initializer.get_peak(x, yhat)
    revised_x = []
    revised_y = []
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
    x_list = k
    plt.yticks(yTick, fontsize=18)
    plt.xticks(np.arange(min(x_list), max(x_list)+1, 1), fontsize=18)

    for prec in prec_list:
        plt.plot(x_list, prec[0], prec[1], label=prec[2], markersize=10)
    plt.legend(fontsize=15,loc=3, framealpha=0.3)
    plt.show()

def plot_behaviors(t):
    with open('behavior_type.json') as f:
        interval = json.load(f)

    bins = np.arange(min(interval[t]) - 1, max(interval[t]) + 1, 5)
    sns.distplot(interval[t], hist=True, kde=True,
             bins=bins, color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
    plt.xlabel('Error of start time', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()
def plot_trends():

    with open('twitch.json') as f:
        data = json.load(f)

    all_videos = []
    chats = []
    viewers = []
    recorded = []
    end = datetime(2018, 10, 29, tzinfo=timezone(timedelta(hours=0)))
    n = 0
    for vl in data.values():
        for v in vl:
            n += 1
            if float(v['chat_number']) / (float(v['length']) / 3600) > 100:
                chats.append(float(v['chat_number']) / (float(v['length']) / 3600))
                viewers.append(v['views'])
                recorded.append(v['recorded'])
    fig1, ax1 = plt.subplots(figsize=(7,3.5))
    # f = plt.figure()
    # plt.figure(1)
    bins = 10 ** np.linspace(np.log10(100), np.log10(25000), 100)
    ax1.set_xscale('log')
    # n, bins, patches = plt.hist(chats, bins=bins)

    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)

    counts, bin_edges = np.histogram(chats, bins=bins, normed=True)

    cdf = np.cumsum (counts)
    ax1.plot(bin_edges[1:], cdf/cdf[-1] * 100, label='cumulative distribution')
    ax1.plot([500, 500], [0,100], label='threshold')

    p = [500, cdf[29] / cdf[-1] * 100]

    ax1.set_yticks([0, p[1], 40, 60, 80, 100])
    ax1.set_xticks([100, 500, 1000, 10000, 25000])
    ax1.axhline(p[1], linestyle='--', color='k')

    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend(fontsize=15, loc=2)
    ax1.set_xlabel('# of chats/hour', fontsize=21)
    ax1.set_ylabel('Percentage (%)', fontsize=21)
    ax1.set_aspect(aspect=2)
    # fig1.savefig('VideoSummarization/analysis/plots/chat_cdf.pdf', format='pdf', bbox_inches='tight')
    fig2, ax2 = plt.subplots(figsize=(7,3.5))
    f = plt.figure()

    # plt.figure(2)
    bins = 10 ** np.linspace(np.log10(100), np.log10(25000), 50)
    ax2.set_xscale("log")
    ax2.set_aspect(aspect=2)
    counts, bin_edges = np.histogram(viewers, bins=bins, normed=True)
    ax2.xaxis.set_tick_params(labelsize=15)
    ax2.yaxis.set_tick_params(labelsize=15)
    cdf = np.cumsum (counts)
    ax2.plot(bin_edges[1:], cdf/cdf[-1] * 100, label='cumulative distribution')
    ax2.plot([100, 100], [0,100], label='threshold')
    ax2.set_xticks([100, 1000, 10000, 25000])
    ax2.set_yticks([0, 20.0, 40.0, 60.0, 80.0, 100.0])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax2.legend(fontsize=15)
    ax2.set_xlabel('# of viewers', fontsize=21)
    ax2.set_ylabel('Percentage (%)', fontsize=21)
    # fig2.savefig('VideoSummarization/analysis/plots/viewer_cdf.pdf', format='pdf', bbox_inches='tight')
    plt.show()
