import pickle
import numpy as np
import torch
dis1 = torch.load('dis/resnest101_16.pth')
dis2 = torch.load('dis/resnest269_16.pth')
dis3 = torch.load('dis/1resnest269_16.pth')
g_camids = pickle.load(open('g_camids','rb'))
distmat = (dis1+dis2+dis3)/3
dis_q= distmat[:,:2915]
query_rank = dis_q.topk(10,largest=False)
dis_g = distmat[:,2915:]
rank = dis_g.topk(1, largest=False)
change={}
for i in range(len(rank[0])):
    if rank[0][i]>0.8:
        for j in range(len(query_rank[0][i])):
            if query_rank[0][i][j]<0.7 and rank[0][query_rank[1][i][j]]<rank[0][i]-0.1:
                change[i] = query_rank[1][i][j]
                break
rank_list = []
for k in range(len(rank[1])):
    if k not in change.keys():
        rank_list.append(g_camids[rank[1][k]])
    else:
        if not(g_camids[rank[1][k]] == g_camids[rank[1][int(change[k])]]):
            print(k)
        rank_list.append(g_camids[rank[1][int(change[k])]])
rank_list = np.array(rank_list)
np.savetxt('submit.csv', rank_list, delimiter='\n', fmt='%s')