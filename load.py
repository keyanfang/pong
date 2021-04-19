import numpy as np
import pickle

def unpack(file):
    with open(file,'wb') as openfile:
        dict=pickle.load(openfile)
    return dict


def load():
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for each_data,each_lable in zip(unpack('data.txt')[b'data'],unpack('data.txt')[b'lable']):
        x1.append(each_data)
        y1.append(each_lable)
    x2=x1[:int(len(x1)*0.2)]
    y2 = x1[:int(len(y1) * 0.2)]
    x1 = x1[int(len(x1) * 0.2)]
    y1 = x1[int(len(y1) * 0.2)]
    x1=np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    valueY=set(y1).union(set(y2))
    mapY={}
    for id,each_valueY in valueY:
        mapY[each_valueY]=id
    y1=np.array(mapY[x]for x in y1)
    y2 = np.array(mapY[x] for x in y2)

