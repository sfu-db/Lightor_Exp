import csv
import numpy as np
import math
from utils import time, sec
import os
from collections import OrderedDict
from parameters import FILTER_INTERVAL, SELECTION_THRESHOLD, ERROR_THRESHOLD, GRAPH_LOWER_BOUND_LENGTH, FEATURE_VECTOR

FILTER_INTERVAL = 60
CROWD_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'dataset', 'Crowdsourcing')
INTERACTIONS = []
with open(os.path.join(CROWD_DATA_PATH, 'interaction.csv'), 'r', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    INTERACTIONS = [r for r in spamreader]

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
    def _transfer(self, time_list):
        re = []
        i = 0
        while (i < len(time_list) - 1):
            if time_list[i + 1] - time_list[i] <= 1.5 and time_list[i + 1] > time_list[i]:
                if len(re) == 0 or re[-1].state != "PL":
                    re.append(PL(time_list[i], 0))
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
                visit_dic[movie_id][visit_id] = self._transfer(visit_dic[movie_id][visit_id])
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
