#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy
from math import *
import copy
import pywt


# In[17]:


def find_corner(I, w_size, threshold, gauss_filter):
    Y = numpy.shape(I)[0]
    X = numpy.shape(I)[1]
    changeX = 0;
    changeY = 0;
    print(Y)
    print(X)
    cornerX = []
    cornerY = []
    R_vals = []
    eig1 = []
    eig2 = []
    for y in range(Y-1):
        for x in range(X-1):
            Ix_sq = 0; Iy_sq = 0; IxIy = 0;
            for i in range(w_size):
                for j in range(w_size):
                    if(y+i-w_size/2>=0 and y+i-w_size/2<Y and x+j-w_size/2>=0 and x+j-w_size/2<X):
                        if(x+1+j-w_size/2 < X and x+1+j-w_size/2 >= 0):
                            changeX = (int(I[y+i-w_size/2][x+1+j-w_size/2]) - int(I[y+i-w_size/2][x+j-w_size/2]))
                            #if(changeX < 0):
                            #    changeX = -changeX;
                            #print(I[y+i-w_size/2][x+1+j-w_size/2] , "" , I[y+i-w_size/2][x+j-w_size/2])
                            #print("changeX:" , changeX)                        
                            Ix_sq = Ix_sq + gauss_filter[i][j]*changeX**2
                            #print("Ix_sq:" , Ix_sq)

                        if(y+1+i-w_size/2 < Y and y+1+i-w_size/2 >= 0):
                            changeY = (int(I[y+i+1-w_size/2][x+j-w_size/2]) - int(I[y+i-w_size/2][x+j-w_size/2]))
                            #if(changeY < 0):
                            #    changeY = -changeY;
                            Iy_sq = Iy_sq + gauss_filter[i][j]*changeY**2
                            IxIy = IxIy + changeX*changeY*gauss_filter[i][j];
                            
            det = Ix_sq*Iy_sq - (IxIy)**2;
            tr = Ix_sq + Iy_sq
            M = numpy.zeros([2,2])
            M[0][0] = Ix_sq; M[0][1] = IxIy
            M[1][0] = IxIy ; M[1][1] = Iy_sq
            eig_val = numpy.linalg.eig(M)
            k = 0.04
            R = eig_val[0][0]*eig_val[0][1] - k*((eig_val[0][0] + eig_val[0][1])**2)
            #if(Ix_sq > threshold and Iy_sq > threshold):\n",
            #print(\"R\", R, \"x\", x , \"y\" , y)\n",
            if(eig_val[0][0] > threshold and eig_val[0][1] > threshold):
                R_vals.append(R)
                cornerX.append(x)
                cornerY.append(y)
                eig1.append(eig_val[0][0])
                eig2.append(eig_val[0][1])
    return cornerX, cornerY, R_vals , eig1, eig2;


# In[135]:


I = cv2.imread('chess.png',1)
I_gray = cv2.imread('chess.png',0)


# In[136]:


filter_size = 7;
sigma = 1.5
normalize = 0;
gauss_filter = numpy.zeros([filter_size,filter_size])


# In[137]:


for y in range(filter_size):
    for x in range(filter_size):
            gauss_filter[y][x] = pow(2*pi,-0.5)*pow(sigma,-1)*exp(-((x-filter_size/2)**2 + (y-filter_size/2)**2)/(2*(sigma**2)))
            normalize = normalize + gauss_filter[y][x]


# In[147]:


Y = numpy.shape(I_gray)[0]
X = numpy.shape(I_gray)[1]
print(Y , X)
I_tran = numpy.zeros([numpy.shape(I_gray)[1], numpy.shape(I_gray)[0]])


# In[149]:


#ROTATION CLK 90
for x in range(X):
    for y in range(Y):
        I_tran[X-1-x][Y-1-y] = I_gray[y][x];
cv2.imwrite('chess_tran.png' , I_tran)
Y_new = numpy.shape(I_tran)[0]
X_new = numpy.shape(I_tran)[1]
print(Y_new, X_new)


# In[161]:


I_comp = numpy.zeros([Y/2, X/2])
#COMPRESSING
for y in range(Y/2):
    for x in range(X/2):
        I_comp[y][x] = I_gray[2*y][2*x]
cv2.imwrite('chess_comp.png', I_comp) 
Y_new = numpy.shape(I_comp)[0]
X_new = numpy.shape(I_comp)[1]
print(Y_new, X_new)


# In[171]:


I_in = copy.deepcopy(I_comp)
threshold = 10**(-2)
cornerX, cornerY, R_vals , eig1, eig2 = find_corner(I_in, filter_size, threshold, gauss_filter)


# In[172]:


print(len(cornerY))


# In[173]:


print(max(eig1) , max(eig2))


# In[174]:


I_comp3D = numpy.zeros([Y_new, X_new, 3])
for y in range(Y_new):
    for x in range(X_new):
        I_comp3D[y][x][:] = I_comp[y][x]


# In[175]:


I_out = copy.deepcopy(I_comp3D)
for i in range(len(cornerX)):
    I_out[cornerY[i]][cornerX[i]][0] = 0
    I_out[cornerY[i]][cornerX[i]][1] = 255
    I_out[cornerY[i]][cornerX[i]][2] = 0
cv2.imwrite('chess_comp_markerd' + str(threshold) + '.png' , I_out)    


# In[ ]:





# In[ ]:




