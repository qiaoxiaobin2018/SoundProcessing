import os
import time
import csv
import random

import numpy as np

# =================windows===========================
load_file_list = "D:\Python_projects\soundProcessing\src/test_list_iden.txt" # train_128_mfcc_for_veri.txt
save_file_list = "D:\Python_projects\SoundProcessing\src/test_pairs_for_veri.csv"
# =================linux===========================
# load_file_list = "/home/longfuhui/shengwenshibie/SoundProcessing/src/train_64_mfcc_for_veri.txt" # train_128_mfcc_for_veri.txt
# save_file_list = "/home/longfuhui/shengwenshibie/SoundProcessing/src/train_64_pairs_for_veri.csv"
save_list = []
list_file_length_1 = 897


# 获取所有文件的路径、标签
iden_list = np.loadtxt(load_file_list, str,delimiter=",")
labels = np.array([int(i[1]) for i in iden_list])
voice_list = np.array([i[0] for i in iden_list])
total_len = len(labels)


def write_csv(pre,last,lable):
    save_list.append([pre, last,lable])


'''
first 是下标
'''


def create_pos_pairs(first):
    write_csv(voice_list[first], voice_list[first+1], 1)
    write_csv(voice_list[first], voice_list[first+2], 1)


def create_neg_pairs(first,list):
    write_csv(voice_list[first],voice_list[list[0]],0)
    write_csv(voice_list[first], voice_list[list[1]],0)


# print(labels[0])
# print(voice_list[0])


def gene_pairs():
    '''
    生成训练对 和 标签
    :return: null
    '''
    '''
    每条数据生成正反例 各 2 对
    对正例，取其下面 2 条数据
    对反例，随机 2 条
    '''


    '''
    先全部遍历，找出每块的边界
    '''
    boundary = {}
    pre_lab = 0
    start = end = 0
    for i in range(total_len):
        this_lab = labels[i]
        # print(i,this_lab)
        if this_lab != pre_lab:
            end = i - 1
            boundary[pre_lab] = [start,end]
            pre_lab = this_lab
            start = i

    print(boundary)
    # print(boundary[0][1])


    '''
    遍历每条数据，添加正反例
    '''
    block = 0
    test_end = boundary[block][1]
    jump = -1
    for i in range(total_len):
        if jump > 0:
            jump -= 1
            continue

        '''
        生成 2 个正例，
        如果当前 i + 2 > test_end，跳至下一个 block
        '''
        if i + 2 > test_end:
            block += 1
            if block > len(boundary) - 1:
                break
            test_end = boundary[block][1]
            jump = 1
            continue
        else:
            # print("=====================")
            create_pos_pairs(i)


        '''
        生成 2 个反例，
        随机选取，不等于当前 lable 即可
        '''
        list = []
        while len(list) < 2:
            intt = random.randint(0,list_file_length_1)
            if labels[intt] != labels[i]:
                list.append(intt)
        create_neg_pairs(i,list)


        # '''
        # 生成 2 个正例，
        # 如果当前 i + 2 > test_end，跳至下一个 block
        # '''
        # if i+2 > test_end:
        #     block += 1
        #     if block > len(boundary) - 1:
        #         break
        #     test_end = boundary[block][1]
        #     jump = 1
        # else:
        #     # print("=====================")
        #     create_pos_pairs(i)

    '''
    写入文件
    '''
    with open(save_file_list, 'a', newline='') as f:
        csv_writer = csv.writer(f,delimiter=' ')
        # 添加标题
        csv_head = ['lable','file1', 'file2']
        csv_writer.writerow(csv_head)
        # 添加内容
        for l in save_list:
            csv_content = [l[2],l[0], l[1]]
            csv_writer.writerow(csv_content)

        # 关闭文件
        f.close()





if __name__ == '__main__':
    gene_pairs()

