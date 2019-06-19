import sys
import os
import numpy as np
from utils import sec, time
from copy import deepcopy

ROOT_PATH = os.path.dirname(os.path.realpath('__file__'))


def evaluate_initializer(k, file_name, dataset):
    path = os.path.join(ROOT_PATH, 'baseline_results', file_name)
    file_list = []
    try:
        file_list = os.listdir(path)
    except:
        print('No such file')
        return
    prec = [0] * k
    remaining = {}
    for f in file_list:
        original_data = np.load(os.path.join(path, f))
        data = [[int(original_data[0][d] // 30), (1 - original_data[2][d]) - (1 - 2 * original_data[2][d]) * original_data[-1][d]] for d in range(len(original_data[0]))]
        data = sorted(data, key=lambda x:-x[-1])
        pred = []
        while len(pred) < k and len(data) > 0:
            flag = True
            cur = data[0][0]
            del data[0]
            for p in pred:
                if abs(cur - p) < 120:
                    flag = False
                    break
            if flag:
                pred.append(cur)
        for i in range(1, k + 1):
            pre = pred[:i]
            prec[i - 1] += len([p for p in pre if p in dataset.start_gt[f.split('.')[0]]])
        remaining[f] = [p for p in pred[:10] if p not in dataset.start_gt[f.split('.')[0]]]
    return [prec[i] / ((i + 1) * len(file_list)) for i in range(k)], remaining

def evaluate_end_to_end(k, file_name, dataset):
    path = os.path.join(ROOT_PATH, 'baseline_results', file_name)
    file_list = []
    try:
        file_list = os.listdir(path)
    except:
        print('No such file')
        return
    start_prec = 0
    end_prec = 0
    remaining = {}
    for f in file_list:
        original_data = np.load(os.path.join(path, f))
        data = [[int(original_data[0][d] // 30), (1 - original_data[2][d]) - (1 - 2 * original_data[2][d]) * original_data[-1][d]] for d in range(len(original_data[0]))]
        frames = deepcopy(data)
        data = sorted(data, key=lambda x:-x[-1])
        pred = []
        start = []
        end = []
        while len(pred) < k and len(data) > 0:
            flag = True
            cur = data[0][0]
            del data[0]
            for p in pred:
                if abs(cur - p) < 120:
                    flag = False
                    break
            if flag:
                pred.append(cur)
        remaining[f] = []
        for i in pred:
            cur = i + 1
            while cur <= len(frames) and frames[cur][1] > 0.5:
                cur += 1
            end.append(cur - 1)
            cur = i
            while cur >= 0 and frames[cur][1] > 0.5:
                cur -= 1
            start.append(cur + 1)
        for i in range(5):
            if start[i] not in dataset.start_gt[f.split('.')[0]] or end[i] not in dataset.end_gt[f.split('.')[0]]:
                remaining[f].append([start[i], end[i]])

        start_prec += len([p for p in start if p in dataset.start_gt[f.split('.')[0]]])
        end_prec += len([p for p in end if p in dataset.end_gt[f.split('.')[0]]])

    return (start_prec / (k * len(file_list))), (end_prec / (k * len(file_list))), remaining