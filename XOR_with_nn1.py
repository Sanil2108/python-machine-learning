#only forward propagation

import numpy as np

#The weights vectors for each perceptron
#for layer 2

t20=[
    1,0,0
]
t30=[
    1,0,0,0,0
]

t21=[
    0,1,0
]
t22=[
    0.5,
    -0.5,
    0
]
t23=[
    0.5,
    0,
    -0.5
]
t24=[
   0,0,1
]

#for layer 3
t31=[
    -6,
    4,
    0,
    4,
    0
]

t32=[
    -6,
    0,
    4,
    0,
    4
]

#for layer 4
t41=[
    -2,
    4,
    4
]

a1=[[1, 0, 1]]
a2=[[0]*5]
a3=[[0]*3]
a4=[[0]*1]

a=[a1, a2, a3, a4]

t2=[t20, t21, t22, t23, t24]
t3=[t30, t31, t32]
t4=[t41]

t=[t2, t3, t4]

#input
a[0]=[[1,0,0]]
for j in range(1, 4):
    #for layer j only
    for i in range(len(a[j][0])):
        if(np.array(np.matrix(a[j-1][0])*np.transpose(np.matrix(t[j-1][i])))[0][0]<=0):
            a[j][0][i]=0
        else:
            a[j][0][i]=1

print(a[3][0][0])