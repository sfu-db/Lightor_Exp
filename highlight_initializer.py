import csv
import datetime
from pprint import pprint
from operator import itemgetter
from sklearn.cluster import KMeans
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import math
from sklearn.model_selection import cross_val_score
import sys
import numpy as np
import scipy.signal
import scipy.stats as stats
from copy import deepcopy
import numpy as np
from math import factorial
from sklearn.externals import joblib
from utils import time, sec

ROOT_PATH = os.path.dirname(os.path.realpath('__file__'))
SIMILAR_WORDS = {"lul":"lol", "rofl":"lol", "lmao":"lol"}
FIGURE_NUMBER = 1
random.seed(1)


def get_peak(x, y):
    """
    Used to detect peak in smoothed curve of chat messages histogram
    """
    re = []
    for i in range(len(x)):
        if  i == 0 and y[i] > y[i + 1]:
            re.append([x[i], y[i]])
        elif i < len(x) - 1 and y[i] > y[i + 1] and y[i] > y[i - 1]:
            re.append([x[i], y[i]])
    p = max(re[::-1], key=lambda x: x[1])
    return p

def scanl(f, base, l):
    for x in l:
        base = f(base, x)
        yield base


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


class DataLoader:
    """
    Used to load dataset and process data
    """
    def __init__(self, dataset_name, dataset_type, **para):

        path = os.path.join(ROOT_PATH, 'dataset', dataset_name, dataset_type)
        chat_path = os.path.join(path, 'chat')
        true_label_path = os.path.join(path, 'labels')
        self.chat = {}
        self.true_labels = {}
        self.sliding_windows = {}
        self.predicted_labels = {}
        self.expanded_labels = {}
        self.sorted_sliding_windows = {}
        self.file_list = []
        self.gt = {}
        self.true_examples = []
        self.false_examples = []
        chat_filename_list = os.listdir(chat_path)

        for f in chat_filename_list:
            split_info = os.path.splitext(f)[0].split()
            # print (split_info)
            if split_info[0] == '.DS_Store' or split_info[0] == '.ipynb_checkpoints':
                continue
            key = split_info[0]
            start_time = int(int(split_info[1]) / 1000)
            file_path = os.path.join(chat_path, f)
            chat_log = list(csv.reader(open(file_path), delimiter='\t'))
            for c in chat_log:
                c[5] = int(int(c[5]) / 1000 - start_time)
            chat_log = sorted(chat_log, key=lambda x:x[5])
            self.chat[key] = chat_log
            self.file_list.append(key)

        try:
            for f in self.file_list:
                file_path = os.path.join(true_label_path, f + '.csv')
                self.true_labels[f] = list(csv.reader(open(file_path), delimiter=' '))
                gt = []
                for j in self.true_labels[f]:
                    gt += list(range(sec(j[1]) - 10, sec(j[2]) + 1))
                self.gt[f] = gt
        except:
            print ("no labels")
        self.get_top_n(para['num'], para['latency'], para['latency'], para['interval_time'])

        for f in self.chat:
            sliding_windows = self.sliding_windows[f]
            for s in sliding_windows:
                fea_vec = [s[4], s[5], s[6]]
                if time(s[0]) in [i[0] for i in self.true_labels[f]]:
                    fea_vec.append('T')
                    self.true_examples.append(fea_vec)
                else:
                    fea_vec.append('F')
                    self.false_examples.append(fea_vec)

        train_pos_num = int(len(self.true_examples) * 1)
        train_neg_num = train_pos_num
        random.shuffle(self.true_examples)
        random.shuffle(self.false_examples)
        self.train_set = []
        self.train_set += self.true_examples[:train_pos_num]
        self.train_set += self.false_examples[:train_neg_num]
        self.train_x = [i[:-1] for i in self.train_set]
        self.train_y = [i[-1] for i in self.train_set]



    def get_top_n(self, num, pre_latency, suc_latency, interval_time):
        for f in self.chat:
            chat_list = self.chat[f]
            sliding_windows = {}
            max_time = chat_list[-1][5]
            for i in range(0, max_time - interval_time):
                sliding_windows[i] = [i, i + interval_time, 0]
            for c in chat_list:
                t = c[5]
                for i in range(max(t - interval_time, 0), min(t + 1, max_time - interval_time)):
                    sliding_windows[i][2] += 1
            sliding_window_values = sliding_windows.values()
            sliding_window_values = sorted(sliding_window_values, key=lambda x:-x[2])
            merged_sliding_windows = self._expand(f, max_time, sliding_window_values, num, pre_latency, suc_latency, chat_list)
            self.sliding_windows[f] = merged_sliding_windows

    def _expand(self, key_name, max_time, sliding_window_values, num, pre_latency, suc_latency, chat_list):
        chosen_windows = []
        while len(chosen_windows) < num and len(sliding_window_values) > 0:
            cur = sliding_window_values[0]
            cur[0] = max(cur[0] - pre_latency, 0)
            cur[1] = min(cur[1] + suc_latency, max_time)
            flag = True
            del sliding_window_values[0]
            for h in chosen_windows:
                if not (cur[0] > h[1] or cur[1] < h[0]):
                    flag = False
                    break
            if flag:
                chosen_windows.append(cur)
        for row in chosen_windows:
            chat_num = 0
            chat = []
            for c in chat_list:
                if c[5] >= row[0] and c[5] <= row[1] and len(c[3]) > 0 and c[3][0] != '@':
                    chat.append([c[3], c[2], c[5]])
                    chat_num += 1
                elif c[5] > row[1]:
                    break
            row.append(chat)
            row.append(chat_num)

        chosen_windows = [r for r in chosen_windows if r[-1] != 0]

        self._get_features(chosen_windows)
        self._normalization(chosen_windows, [2, 4, 5, 6])
        chosen_windows = sorted(chosen_windows, key=lambda x:x[0])
        return chosen_windows

    def _get_features(self, chosen_windows):
        for r in range(len(chosen_windows)):
            row = chosen_windows[r]
            word_set = set()
            word_list = []
            vector_list = []
            len_list = []
            for chat in row[3]:
                c = chat[0]
                len_list.append(len(c))
                word = c.lower().split()
                for i in range(len(word)):
                    if word[i] in SIMILAR_WORDS:
                        word[i] = SIMILAR_WORDS[word[i]]
                word_list.append(word)
                word_set = word_set.union(set(word))
            row.append(sum(len_list) / float(len(len_list)))

            for c in word_list:
                vector = []
                for w in word_set:
                    if w in c:
                        vector.append(1)
                    else:
                        vector.append(0)
                vector_list.append(vector)
            vector_list = np.array(vector_list)
            kmeans = KMeans(n_clusters=1, random_state=0).fit(vector_list)
            center = kmeans.cluster_centers_[0]
            similarity = self._sim(vector_list, center)
            row.append(similarity)

    def _sim(self, vector_list, center):
        n = 0
        l = len(vector_list[0])
        l_list = []
        for v in vector_list:
            m = 0
            for i in range(len(v)):
                m += pow((float)(v[i]) - center[i], 2) * 1
            l_list.append(float(math.sqrt(m / float(l))))
        l_list = sorted(l_list)
        return l_list[int(len(l_list) / 2)]

    def _normalization(self, windows, vec):
        for j in vec:
            j_list = [i[j] for i in windows]
            min_j = min(j_list)
            max_j = max(j_list)
            for k in windows:
                k[j] = float(k[j] - min_j) / float(max_j - min_j)


def train(dataset, fea_vec):
    clf = LogisticRegression(C=10)
    X = []
    for i in dataset.train_x:
        X.append([i[j] for j in fea_vec])
    clf.fit(X, dataset.train_y)
    return clf


def apply_model(model, test_data, interval, fea_vec):
    file_list = list(test_data.sliding_windows.keys())
    for f in file_list:
        temp = [[i[0]] for i in test_data.sliding_windows[f]]
        features = [[i[k] for k in fea_vec] for i in test_data.sliding_windows[f]]
        result = model.predict(features)
        result_proba = model.predict_proba(features)
        for i, j in enumerate(temp):
            j.append(result_proba[i][1])
        for i, j in enumerate(test_data.sliding_windows[f]):
            j.append(result_proba[i][1])
        temp = sorted(temp, key=lambda x: -x[-1])
        test_data.sorted_sliding_windows[f] = temp
        re = []
        i = 0
        while i < len(temp):
            cur = temp[i]
            flag = True
            for r in re:
                if abs(r[0] - cur[0]) < interval:
                    flag = False
                    break
            i += 1
            if flag:
                re.append(cur)
        test_data.predicted_labels[f] = re

class Adjustment:
    """
    Used for doing adjustment
    """
    def __init__(self, data, **kwargs):
        self.data = data
        self.const = None
        self.labels_by_video = {}
        self.all_labels = []
        self.gt = []

        for f in self.data.true_labels:
            self.labels_by_video[f] = [[l[0], sec(l[1]), sec(l[2])]
                                       for l in self.data.true_labels[f] if l[0] != '000-00']

            for t in self.labels_by_video[f]:
                pre, expand_info = self._get_pre_windows(f, self.data, t[0], **kwargs)
                t.append(expand_info)
                self.all_labels.append(t)
                self.gt += list(range(t[1] - 10, t[2] + 1))

    def aug_max(self, t):
        mean_gap = int(sum([(i[3] - i[2] + (i[2] - i[1] + 10) / 2) for i in t]) / len(t))
        start = -40
        end = mean_gap + 40
        scores = []
        for i in range(start, end):
            score = 0
            for j in t:
                if (j[3] - i >= j[1] - 10 and j[3] - i <= j[2]):
                    score += 1
            scores.append([i, score])
        # print('scores', scores)
        max_score = max(scores, key=lambda x:x[1])
        # print('precision:', max_score[1] / float(len(t)))
        return max_score


    def get_avg_length(self):
        len_list = []
        for f in self.data.true_labels:
            len_list += [sec(i[2]) - sec(i[1]) for i in self.data.true_labels[f]]
        return int(sum(len_list) / len(len_list))

    def train(self):
        expand_const = self.aug_max(self.all_labels)
        avg_length = self.get_avg_length()
        self.const = [expand_const[0], avg_length]

    def apply(self, data):
        data.avg_length = self.const[1]
        for f in data.predicted_labels:
            for t in data.predicted_labels[f]:
                if len(t) > 3:
                    t[3] = t[2] - self.const[0]
                else:
                    t.append(t[2] - self.const[0])

    def generate_peak(self, data, **kwargs):
        labels_by_video = {}
        for f in data.predicted_labels:
            for t in data.predicted_labels[f]:
                pre, expand_info = self._get_pre_windows(f, data, time(t[0]), **kwargs)
                t.append(expand_info)

    def _get_pre_windows(self, video_name, data, time_stamp, **kwargs):
        chat_len_threshold = 25
        pre_windows = []
        sliding_windows_list = data.sliding_windows[video_name]
        comments = data.chat[video_name]
        for i in range(len(sliding_windows_list)):
            if sliding_windows_list[i][0] == sec(time_stamp):
                if i == 0:
                    pre_windows = deepcopy([sliding_windows_list[i]])
                else:
                    pre_windows = deepcopy(sliding_windows_list[i - 1 : i + 1])
                    curr_win = sliding_windows_list[i]
                    for c in comments:
                        if c[5] > pre_windows[0][1] and c[5] < pre_windows[1][0]:
                            pre_windows[0][3].append([c[3], c[2], c[5]])
                        elif c[5] >= pre_windows[1][0]:
                            break
                break
        expander = Expander(sliding_windows_list)
        expand_info = expander.expand(pre_windows, chat_len_threshold, **kwargs)


        return pre_windows, expand_info


class Expander:
    def __init__(self, sliding_windows_list):
        self.sliding_windows_list = sliding_windows_list
    def expand(self, windows, l, **kwargs):
        re = self._expand_by_peak(windows, kwargs['bin_size'], kwargs['win_size'], kwargs['polyorder'])
        return re


    def _bin(self, size, windows):
        re = {}
        time_list = []
        start = 0
        end = 0
        if len(windows) > 1:
            time_list = [j[2] for j in windows[0][3] + windows[1][3]]
            start = windows[0][0]
            end = windows[1][1]
        else:
            time_list = [j[2] for j in windows[0][3]]
            start = windows[0][0]
            end = windows[0][1]
        k = start
        while k <= end:
                re[k] = 0
                k += size
        for t in time_list:
            for b in re:
                if t > b and t <= b + size:
                    re[b] += 1
                    break
        re = sorted(re.items(), key=lambda x: x[0])
        return re

    def _expand_by_peak(self, windows, bin_size, win_size, polyorder):
        re = self._bin(bin_size, windows)
        x = [j[0] for j in re]
        y = np.asarray([j[1] for j in re])
        yhat = savitzky_golay(y, win_size, polyorder)
        p = get_peak(x, yhat)
        return p[0]
