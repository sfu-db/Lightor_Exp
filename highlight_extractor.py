import csv
import numpy as np
import math
from utils import time, sec
import os
from collections import OrderedDict
from parameters import *
import copy
import highlight_initializer
import random   

FILTER_INTERVAL = 60
CROWD_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'dataset', 'Crowdsourcing')
INTERACTIONS = []
with open(os.path.join(CROWD_DATA_PATH, 'interaction.csv'), 'r', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    INTERACTIONS = [r for r in spamreader]

def check_prec(d, dataset):
    start = []
    end = []
    for v in d:
        f = DOTA2_MOVIE_NAME_BY_ID[v]
        if d[v][0] in dataset.start_gt[f]:
            start.append(1)
        else:
            start.append(0)
        if d[v][1] in dataset.end_gt[f]:
            end.append(1)
        else:
            end.append(0)
    return np.mean(start), np.mean(end)

def transfer(time_list):
    re = []
    i = 0
    while (i < len(time_list) - 1):
        if time_list[i + 1] - time_list[i] <= 1.5 and time_list[i + 1] > time_list[i]:
            if len(re) == 0 or re[-1].state != "PL":
                re.append(PL(time_list[i], time_list[i + 1]))
            else:
                re[-1].change_end(time_list[i])
        elif time_list[i + 1] < time_list[i]:
            if len(re) != 0 and re[-1].state == "PL":
                re[-1].change_end(time_list[i])
            re.append(SB(time_list[i + 1]))
        elif time_list[i + 1] - time_list[i] > 1.5:
            if len(re) !=0 and re[-1].state == "PL":
                re[-1].change_end(time_list[i])
            re.append(SF(time_list[i + 1]))
        i += 1
    return re


class PL:
    def __init__(self, s, e):
        self.start = s
        self.end = e
        self.state = "PL"
        self.duration = e - s
    def change_end(self, e):
        self.end = e
        self.duration = self.end - self.start
    def __str__(self):
        return 'PL: start={0} {1}, end={2} {3}, duration={4}'.format(self.start, time(self.start), self.end, time(self.end), self.end - self.start)
    def __repr__(self):
        return "PL"
    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        return (self.start, self.end) == (other.start, other.end)


class SB:
    def __init__(self, s):
        self.seek = s
        self.state = "SB"
    def __str__(self):
        return 'SB: seek={0} {1}'.format(self.seek, time(self.seek))
    def __repr__(self):
        return "SB"


class SF:
    def __init__(self, s):
        self.seek = s
        self.state = "SF"
    def __str__(self):
        return 'SF: seek={0} {1}'.format(self.seek, time(self.seek))
    def __repr__(self):
        return "SF"

class Highlight_Extractor:
    def __init__(self, original_data):
        self.feature_vector = FEATURE_VECTOR
        self.data_lineage = original_data
        self.assignments = {}

    def _overlap(self, p1, p2):
        return not (p1[0].start >= p2[0].end or p2[0].start >= p1[0].end)

    def _construct_graph(self, pl_list, threshold):
        graph = {}
        filtered_pl_list = [pl for pl in pl_list if pl[0].duration > threshold]
        filtered_pl_list = sorted(filtered_pl_list, key=lambda x: x[1])
        for pl1 in filtered_pl_list:
            graph[pl1[0]] = [pl1]
            for pl2 in filtered_pl_list:
                if self._overlap(pl1, pl2):
                    graph[pl1[0]].append(pl2)
        max_ = [0, []]
        for k in graph:
            if len(graph[k]) > max_[0]:
                max_ = [len(graph[k]), graph[k]]
        return max_[1]

    def _cos_sim(self, a, b):
        m = sum([a[i] * b[i] for i in range(len(a))])
        n = math.sqrt(sum([a[i] * a[i] for i in range(len(a))])) * math.sqrt(sum([b[i] * b[i] for i in range(len(b))]))
        return m / n
    def _clustering(self, f):
        if self._cos_sim(f, self.feature_vector[0]) > self._cos_sim(f, self.feature_vector[1]):
            return 1
        return 2

    def _analyze(self, labels):
        selection_threshold = SELECTION_THRESHOLD
        error_threshold = ERROR_THRESHOLD

        self.data_lineage['remaining'] = {}
        d = labels['remaining']
        for m in d:
            if len(d[m]) < 3:
                self.data_lineage['remaining'][m] = d[m]
            else:
                interaction_list = d[m][-3:]
                start_list = [i[0] for i in interaction_list]
                end_list = [i[1] for i in interaction_list]
                if len(set(start_list)) == 1:
                    if -1 in end_list:
                        end_list.remove(-1)
                    if max(end_list) - min(end_list) <= selection_threshold:
                        self.data_lineage['selected'][m] = d[m] + [[start_list[-1], sorted(end_list)[int(len(end_list) / 2)]]]
                    elif max(end_list) - min(end_list) >= error_threshold:
                        self.data_lineage['filtered'][m] = d[m] + [[-1, -1]]
                    else:
                        self.data_lineage['remaining'][m] = d[m]
                elif start_list[1] - start_list[0] > error_threshold or start_list[2] - start_list[1] > error_threshold:
                    self.data_lineage['filtered'][m] = d[m] + [[-1, -1]]
                else:
                    self.data_lineage['remaining'][m] = d[m]

    def _estimate(self, k, user_data, cluster_label, label):
        filtered_list = []
        all_play = []
        for assignment_id in user_data:
            first_play = user_data[assignment_id][0]
            if abs(first_play.start - label[0]) <= 1:
                if len(user_data[assignment_id]) > 1:
                    filtered_list += user_data[assignment_id][1: len(user_data[assignment_id])]
            all_play += user_data[assignment_id]
        all_end = sorted([int(i.end) for i in all_play if i.end > label[0]])
        estimated_start = label[0]
        estimated_end = all_end[int(len(all_end) / 2)]
        if cluster_label == 1:
            filtered_list = [i for i in filtered_list if i.end >= label[0]]
            if len(filtered_list) > 0:
                end_list = sorted([int(i.end) for i in filtered_list])
                estimated_end = int(end_list[int(len(end_list) / 2)])
                start_list = sorted([int(i.start) for i in filtered_list if i.start < estimated_end and int(i.start) >= label[0]])
                estimated_start = int(start_list[int(len(start_list) / 2)]) if len(start_list) > 0 else label[0]
        else:
            estimated_start = label[0] - 20
            estimated_end = -1
        return [estimated_start, estimated_end]

    def _process_one_iter(self, answer_file_name):
        assignments_list = {}
        visit_dic = {}
        labels = {}
        labels['remaining'] = self.data_lineage['remaining']
        current_label = {}
        for t in labels:
            for k in labels[t]:
                current_label[k] = labels[t][k][-1]
        with open(os.path.join(CROWD_DATA_PATH, 'answers', answer_file_name), 'r', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                assignments_list[row[0]] = row[2]
        for k in INTERACTIONS:
            if k[1] in assignments_list:
                if  str(k[2]) in current_label and float(k[0]) >= current_label[str(k[2])][0] - FILTER_INTERVAL and float(k[0]) <= current_label[str(k[2])][0] + FILTER_INTERVAL:
                    if str(k[2]) not in visit_dic:
                        visit_dic[str(k[2])] = {}
                    if str(k[1]) not in visit_dic[str(k[2])]:
                        visit_dic[str(k[2])][k[1]] = [float(k[0])]
                    else:
                        visit_dic[str(k[2])][k[1]].append(float(k[0]))
        for movie_id in list(visit_dic):
            for visit_id in list(visit_dic[movie_id]):
                flag = True
                visit_dic[movie_id][visit_id] = transfer(visit_dic[movie_id][visit_id])
                for op in visit_dic[movie_id][visit_id]:
                    if op.state == 'PL'and op.duration >= 100:
                        flag = False
                        break
                if flag == False:
                    del visit_dic[movie_id][visit_id]

        pl_list = OrderedDict()
        graph_list = OrderedDict()
        for movie_id in visit_dic:
            pl_list[movie_id] = []
            for visit_id in visit_dic[movie_id]:
                pl_list[movie_id] += [[i, visit_id] for i in visit_dic[movie_id][visit_id] if i.state == "PL"]
            # del pl_list[movie_id][0]
            graph_list[movie_id] = self._construct_graph(pl_list[movie_id], GRAPH_LOWER_BOUND_LENGTH)

        processed_data = {}
        for k in current_label:
            data_by_assignment = {}
            data = [0, 0, 0]
            for c in graph_list[k]:
                if c[1] not in data_by_assignment:
                    data_by_assignment[c[1]] = [c[0]]
                else:
                    data_by_assignment[c[1]].append(c[0])
                if c[0].end <= current_label[k][0]:
                    data[0] += 1
                elif c[0].start < current_label[k][0]:
                    data[1] += 1
                elif c[0].start >= current_label[k][0]:
                    data[2] +=1
            graph_list[k] = data_by_assignment
            processed_data[k] = data

        for k in labels['remaining']:
            new_pos = self._estimate(k, graph_list[k], self._clustering(processed_data[k]), current_label[k])
            labels['remaining'][k].append(new_pos)
        self._analyze(labels)

    def process_all_iterations(self):
        iter = 7
        for i in range(1, iter + 1):
            file_name = 'experiments_dota_{0}.csv'.format(i)
            self._process_one_iter(file_name)


class Score:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.__len__ = end - start + 1
        self.score_arr = np.zeros(self.__len__)

    def pos(self):
        self.score_arr = np.ones(self.__len__)

    def neg(self):
        self.score_arr = np.negative(np.ones(self.__len__))
    


class Baselines:
    def __init__(self, ORIGINAL_DATA):
        self.vid = {}
        self.bf_bound = []
        self.pl_bound = []
        self.original_data = ORIGINAL_DATA

    def _get_inters_per_iter(self, answer_file, ratio):
        assignment_ids = []
        hl_ids = copy.deepcopy(tuple(self.original_data['remaining'].keys()))
        assignments_hl = dict.fromkeys(hl_ids)
        hl_inter = dict.fromkeys(hl_ids)
        with open(answer_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                assignment_ids.append(row[0])
            # print(len(assignment_ids))
        random.shuffle(assignment_ids)
        assignment_ids = assignment_ids[:int(len(assignment_ids) * ratio)]

        for interaction in INTERACTIONS:
            hl_id = interaction[2]
            as_id = interaction[1]
            time_stamp = round(float(interaction[0]))
            
            if as_id in assignment_ids:
                if hl_id in assignments_hl:
                    if assignments_hl[hl_id] is None:
                        assignments_hl[hl_id] = {}
                    else:
                        if as_id in assignments_hl[hl_id]:
                            assignments_hl[hl_id][as_id].append(time_stamp)
                        else:
                            assignments_hl[hl_id][as_id] = [time_stamp]
                else:
                    print("hl not in parameter.py file")

        for hl_id, as_dict in assignments_hl.items():
            if as_dict is not None:
                time_lists = list(as_dict.values())
                inters_list = []
                for time_list in time_lists:
                    inters = transfer(time_list)
                    first_pl_pos = 0
                    if inters:
                        for idx, inter in enumerate(inters):
                            if inter.state == 'PL':
                                first_pl_pos = idx
                                break
                            else:
                                first_pl_pos += 1
                        inters = inters[first_pl_pos:]
                        if len(inters) >= 1:
                            inters_list.append(inters)
                hl_inter[hl_id] = inters_list
            else:
                hl_inter.pop(hl_id)
        return assignments_hl, hl_inter

    def _merge_2_scores(self, score1, score2):
        new_start = min(score1.start, score2.start)
        start_gap = abs(score1.start - score2.start)
        new_end = max(score1.end, score2.end)
        new_score = Score(new_start, new_end)
        if score1.start == new_start:
            new_score.score_arr[:len(score1.score_arr)] += score1.score_arr
            new_score.score_arr[start_gap:start_gap+len(score2.score_arr)] += score2.score_arr
        else:
            new_score.score_arr[:len(score2.score_arr)] += score2.score_arr
            new_score.score_arr[start_gap:start_gap+len(score1.score_arr)] += score1.score_arr
        return new_score

    def _backward_forward_score(self, inters_list):
        start = inters_list[0][0].start
        end = inters_list[0][0].end
        final_score = Score(start, end)
        for i in range(len(inters_list)):
            start = inters_list[i][0].start
            end = inters_list[i][0].end
            score_per_as = Score(start, end)
            for j in range(len(inters_list[i]) - 1):
                if inters_list[i][j].state == 'PL':
                    start = inters_list[i][j].end
                    end = inters_list[i][j+1].seek
                    if inters_list[i][j+1].state == 'SF':
                        curr_score = Score(start, end)
                        curr_score.neg()
                    elif inters_list[i][j+1].state == 'SB':
                        curr_score = Score(end, start)
                        curr_score.pos()
                else:
                    start = inters_list[i][j].seek
                    if inters_list[i][j+1].state == 'SF':
                        end = inters_list[i][j+1].seek
                        curr_score = Score(start, end)
                        curr_score.neg()
                    elif inters_list[i][j+1].state == 'SB':
                        end = inters_list[i][j+1].seek
                        curr_score = Score(end, start)
                        curr_score.pos()
                    else:
                        end = inters_list[i][j+1].start
                        curr_score = Score(start, end)
                score_per_as = self._merge_2_scores(score_per_as, curr_score)
            final_score = self._merge_2_scores(final_score, score_per_as)
        return final_score

    def _play_score(self, inters_list):
        start = inters_list[0][0].start
        end = inters_list[0][0].end
        score = Score(start, end)
        for i in range(len(inters_list)):
            for j in range(len(inters_list[i])):
                if inters_list[i][j].state == 'PL':
                    start = inters_list[i][j].start
                    end = inters_list[i][j].end
                    curr_score = Score(start, end)
                    curr_score.pos()
                    score = self._merge_2_scores(score, curr_score)
        return score
    
    # def _pl_smooth_score(self, some_score):
    #     smoothed_score = sm.nonparametric.lowess(some_score.score_arr, range(some_score.start, some_score.end + 1), frac=0.1 ,return_sorted=True)
    #     return smoothed_score.T

    def get_baseline_score(self, ratio):
        answer_folder = os.path.join(CROWD_DATA_PATH, 'answers')
        exp_files = [os.path.join(answer_folder, i) for i in os.listdir(answer_folder)]
        exp_files.sort()
        exp_files = exp_files[:1]
        bf_bound = []
        pl_bound = []
        for exp in exp_files:
            _ , hl_inter = self._get_inters_per_iter(exp, ratio)
            bf_bound_iter = {}
            pl_bound_iter = {}
            for hl in hl_inter:
                # bf bound
                bf_score_per_hl = self._backward_forward_score(hl_inter[hl])
                bf_score_per_hl.score_arr = highlight_initializer.savitzky_golay(bf_score_per_hl.score_arr, window_size=9, order=4)
                bf_peak_idx = np.argmax(bf_score_per_hl.score_arr)
                bf_peak_pos = range(bf_score_per_hl.start, bf_score_per_hl.end + 1)[bf_peak_idx]
                hl_bound = [max(0, bf_peak_pos - 10), bf_peak_pos + 10]
                bf_bound_iter[hl] = hl_bound
                # pl bound
                pl_score_per_hl_non_smooth = self._play_score(hl_inter[hl])
                pl_score_per_hl = pl_score_per_hl_non_smooth
                pl_score_per_hl.score_arr = highlight_initializer.savitzky_golay(pl_score_per_hl_non_smooth.score_arr, window_size=9, order=4)
                peak_idx = int(np.argmax(pl_score_per_hl.score_arr))
                start_bound_idx = peak_idx
                end_bound_idx = peak_idx
                for i in range(peak_idx, -1, -1):
                    if pl_score_per_hl.score_arr[i] <= pl_score_per_hl.score_arr[start_bound_idx]:
                        start_bound_idx = i
                    else:
                        break
                for j in range(peak_idx, len(pl_score_per_hl.score_arr)):
                    if pl_score_per_hl.score_arr[j] <= pl_score_per_hl.score_arr[end_bound_idx]:
                        end_bound_idx = j
                    else:
                        break
                pl_range = range(pl_score_per_hl.start, pl_score_per_hl.end + 1)
                pl_bound_iter[hl] = [pl_range[start_bound_idx], pl_range[end_bound_idx]]
            # print(bf_bound_iter, pl_bound_iter)
            bf_bound.append(bf_bound_iter)
            pl_bound.append(pl_bound_iter)
            self.bf_bound = bf_bound
            self.pl_bound = pl_bound
        
