import sys
import os
import numpy as np
from highlight_initializer import sec, time

ROOT_PATH = os.path.dirname(os.path.realpath('__file__'))

"""
we relabelled results generated by baseline because it has no "sliding window"
format (which we laballed in dataset) in our evaluation metrics (chat precision).
We tried to make a fair comparison with baseline here.
"""

extra_labels = {
    'attackerdota-2017-07-31-02h34m10s': ['018-11'],
    'attackerdota-2017-07-30-23h34m09s': ['026-08'],
    'moonmeander-2017-07-03-19h17m57s': ['089-09', '082-15'],
    'sing_sing-2017-08-01-09h09m34s': ['002-05' ],
    'sing_sing-2017-08-01-10h09m35s': ['008-58', '043-24'],
    'attackerdota-2017-07-31-01h34m09s': ['035-13', '037-58', '052-29'],
    'moonmeander-2017-07-03-17h17m56s': ['021-34', '068-19'],
    'nalcs_w2d2_DIG_FOX_g3':['016-01', '019-01', ],
    'nalcs_w8d2_TL_P1_g2':['010-57', '019-22', '023-24', '041-48', '011-56'],
    'nalcs_w8d2_TL_P1_g1':['017-31', '023-50'],
    'nalcs_w2d2_DIG_FOX_g2':['026-26'],
    'nalcs_w2d2_P1_NV_g1': ['042-38', '039-42'],
    'nalcs_w8d2_NV_DIG_g1' : ['038-08'],
    'nalcs_w8d2_NV_DIG_g2' : ['048-22', '046-51']
}

def evaluate_baseline(k, file_name, gt):
    path = os.path.join(ROOT_PATH, 'baseline_results', file_name)
    file_list = []
    try:
        file_list = os.listdir(path)
    except:
        print('No such file')
        return
    prec = [0] * k
    for f in file_list:
        data = np.load(os.path.join(path, f))
        data = [[data[0][d] / 30, data[-1][d]] for d in range(len(data[0])) if data[2][d] == 1]
        data = sorted(data, key=lambda x:-x[-1])
        pred = []
        while len(pred) < 10 and len(data) > 0:
            flag = True
            cur = data[0][0]
            del data[0]
            for p in pred:
                if abs(cur - p) < 120:
                    flag = False
                    break
            if flag:
                pred.append(cur)
        all_gt = []
        for i in gt[f.split('.')[0]]:
            all_gt += list(range(sec(i[0]), sec(i[0]) + 26))
        for i in extra_labels[f.split('.')[0]]:
            all_gt += list(range(sec(i), sec(i) + 26))
        for i in range(1, k + 1):
            pre = pred[:i]
            prec[i - 1] += len([k for k in pre if k in all_gt or k + 6 in all_gt])
    return [prec[i] / ((i + 1) * len(file_list)) for i in range(k)]