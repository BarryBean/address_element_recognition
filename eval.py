#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：yusheng 
# albert time:2021/8/22


from collections import Counter
from os.path import join
from codecs import open


def predict_test_acc():
    print('==================predicting test acc===================')
    submit_test = open(
        r'D:\CodePractice\Python\address\ccks2021_task2_baseline\data\outputs\addr_parsing_runid_lex_all_best.txt', 'r',
        encoding='utf8')
    final_test = open(r'D:\CodePractice\Python\address\ccks2021_task2_baseline\data\testtag.txt', 'r',
                      encoding='utf8')
    N = 0
    for line in submit_test:
        line = line.strip()
        if line == '':
            continue
        N = N + 1
        parts = line.split("\001")
        if len(parts[1]) != len(parts[2].split(' ')):
            print(line)
            raise AssertionError(f"请保证句子长度和标签长度一致，且标签之间用空格分隔！ID:{parts[0]} Sent:{parts[1]}")
        elif parts[0] != str(N):
            raise AssertionError(f"请保证测试数据的ID合法！ID:{parts[0]} Sent:{parts[1]}")
        else:
            for tag in parts[2].split(' '):
                if (tag == 'O' or tag.startswith('S-')
                    or tag.startswith('B-')
                    or tag.startswith('I-')
                    or tag.startswith('E-')) is False:
                    raise AssertionError(f"预测结果存在不合法的标签！ID:{parts[0]} Tag:{parts[2]}")

    test_word_lists, test_tag_lists = build_corpus("test")
    pre_tag_lists = build_corpus2(
        r'D:\CodePractice\Python\address\ccks2021_task2_baseline\data\outputs\addr_parsing_runid_lex_no_encoder.txt')
    metrics = Metrics(test_tag_lists, pre_tag_lists)
    metrics.report_scores()  # 打印每个标记的精确度、召回率、f1分数
    metrics.report_confusion_matrix()  # 打印混淆矩阵

    te, be, ue = metrics.cal_error()
    print(te, be, ue)

    submit_test.close()
    final_test.close()


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


def build_corpus(split, data_dir="./data"):
    """读取数据"""
    assert split in ['train', 'dev', 'test', 'extra_test_v2']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split + ".conll"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            # if line != '\r\n':
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    return word_lists, tag_lists


def build_corpus2(data_dir="./data"):
    """读取数据"""

    with open(join(data_dir), 'r', encoding='utf-8') as f:
        tag_list = []
        for line in f:
            if line != '\n':
                test_tags = line.strip().split('\001')[2]
                tag = test_tags.split(' ')
                # word, tag = line.strip('\n').split()
                tag_list.append(tag)
    return tag_list


class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, golden_tags, predict_tags):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        self.te = 0
        self.be = 0
        self.ue = 0

        # 辅助计算的变量
        self.tagset = set(self.golden_tags)
        self.tagset = sorted(self.tagset)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # 计算精确率
        self.precision_scores = self.cal_precision()

        # 计算召回率
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()

    def cal_error(self):
        return self.te, self.be, self.ue

    def cal_precision(self):

        precision_scores = {}
        for tag in self.tagset:
            if self.predict_tags_counter[tag] == 0:
                precision_scores[tag] = 0
                continue
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / self.predict_tags_counter[tag]

        return precision_scores

    def cal_recall(self):

        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / self.golden_tags_counter[tag]
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2 * p * r / (p + r + 1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self):
        # 打印表头
        header_format = '{:>12s}  {:>12} {:>12} {:>12} {:>12}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>15s}  {:>12.4f} {:>12.4f} {:>12.4f} {:>12}'
        # 打印每个标签的 精确率、召回率、f1分数
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        # 计算并打印平均值
        avg_metrics = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))

    def count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1
            else:
                # 判断错误种类：type err。bound err。unknown err。
                gtags = gold_tag.split("-")
                ptags = predict_tag.split("-")
                # 先判断是不是O
                if len(gtags) != 1 and len(ptags) == 1:
                    self.te = self.te + 1
                elif len(gtags) == 1 and len(ptags) != 1:
                    self.te = self.te + 1
                elif len(gtags) != 1 and len(ptags) != 1:
                    if gtags[1] != ptags[1]:
                        self.te = self.te + 1
                    elif gtags[0] != ptags[0]:
                        self.be = self.be + 1
                    else:
                        self.ue = self.ue + 1
        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def report_confusion_matrix(self):
        """计算混淆矩阵"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:
                continue

        # 输出矩阵
        row_format_ = '{:>14} ' * (tags_size + 1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))


if __name__ == '__main__':
    predict_test_acc()
    pass
