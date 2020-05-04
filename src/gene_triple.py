import os
import time
import csv
import random

import numpy as np

# =================windows===========================
load_file_list = "D:\Python_projects\soundProcessing\src/triple_list_rand_5.txt" # train_128_mfcc_for_veri.txt
save_file_list = "D:\Python_projects\SoundProcessing\src/triple_rand_5.csv"
# =================linux===========================
# load_file_list = "/home/longfuhui/shengwenshibie/SoundProcessing/src/train_64_mfcc_for_veri.txt" # train_128_mfcc_for_veri.txt
# save_file_list = "/home/longfuhui/shengwenshibie/SoundProcessing/src/train_64_pairs_for_veri.csv"
save_list = []


# 获取所有文件的路径、标签
iden_list = np.loadtxt(load_file_list, str,delimiter=",")
labels = np.array([int(i[2]) for i in iden_list])
owner_list = np.array([i[0] for i in iden_list])
guest_list = np.array([i[1] for i in iden_list])
total_len = len(labels)


def write_csv(owner,p,n):
    save_list.append([owner,p,n])


'''
first 是下标
'''


def create_this_triple(owner,plist,nlist):
    for i in range(3):
        write_csv(owner,plist[i],nlist[i])






def gene_triple():
    '''
    生成训练对 和 标签
    :return: null
    '''


    for i in range(total_len):
        if i%6 == 0:
            owner = owner_list[i]
            p_list = []
            n_list = []
            for j in range(3):
                p_list.append(guest_list[i+j])
                n_list.append(guest_list[i+3+j])

            create_this_triple(owner,p_list,n_list)





    '''
    写入文件
    '''
    with open(save_file_list, 'a', newline='') as f:
        csv_writer = csv.writer(f,delimiter=',')
        # 添加标题
        csv_head = ['owner','p', 'n']
        csv_writer.writerow(csv_head)
        # 添加内容
        for l in save_list:
            csv_content = [l[0],l[1], l[2]]
            csv_writer.writerow(csv_content)

        # 关闭文件
        f.close()





if __name__ == '__main__':
    gene_triple()

