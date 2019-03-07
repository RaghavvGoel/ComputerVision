#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy
from math import *
import copy
import pywt

# Y = numpy.shape(I_gray)[0]
# X = numpy.shape(I_gray)[1]


# In[5]:


def find_circle(I, r_min, r_max, X, Y):
    R = r_max - r_min + 1;
    hoff_mat = numpy.zeros([Y,X,r_max])
    for y in range(r_min,Y-r_min,1):
        for x in range(r_min,X-r_min,1):
            if(I[y][x] == 255):
                a_prev = 0; b_prev = 0;
                for r in range(r_min,r_max,1):
                      for theta in range(0,360,1): #step = 1
                        #iterations = iterations + 1;
                        a = int(round(x + r*numpy.cos(theta*pi/180)))
                        b = int(round(y + r*numpy.sin(theta*pi/180)))                        
                        #check if white 
                        if(a_prev != a and b_prev != b): #to avoid voting to same points due to round
                            if(a >= 0 and a < X and b >= 0 and b < Y):                        
                                #count = count + 1
                                hoff_mat[b][a][r] += 1;
                                #print("a=", a, " b=",b)
                            a_prev = a; b_prev = b;    
    return hoff_mat;


# In[6]:


def laplacian(I):
	I_lap = copy.deepcopy(I)
	lap_filter = numpy.ones([3,3])
	lap_filter = -lap_filter
	lap_filter[1][1] = 8;
	X = numpy.shape(I)[1]
	Y = numpy.shape(I)[0]
	for y in range(Y):
		for x in range(X):
			temp_sum = 0;
			for i in range(3):
				for j in range(3):
					if(x+j-1<0 or x+j-1>=X  or y+i-1<0 or y+i-1>=Y):
						temp_sum = temp_sum;
					else:
						temp_sum = temp_sum + I[y+i-1][x+j-1]*lap_filter[i][j];
			if(temp_sum < 0):
				I_lap[y][x] = 0;
			else:
				I_lap[y][x] = temp_sum					
	return I_lap


# In[7]:


def gauss_filter(I_gray, filter_size, sigma):
	normalize = 0;
	gauss_filter = numpy.zeros([filter_size,filter_size])
	for y in range(filter_size):
		for x in range(filter_size):
				gauss_filter[y][x] = pow(2*pi,-0.5)*pow(sigma,-1)*exp(-((x-filter_size/2)**2 + (y-filter_size/2)**2)/(2*(sigma**2)))
				normalize = normalize + gauss_filter[y][x]

	#print("gauss_filter")
	#print(gauss_filter)
	print("normalize value: " , normalize)
	Y = numpy.shape(I)[0]
	X = numpy.shape(I)[1]

	I_gauss = copy.deepcopy(I_gray)

	for y in range(Y):
		for x in range(X):
			temp_sum = 0;
			for i in range(filter_size):
				for j in range(filter_size):
					if(x+j-filter_size/2 < 0 or x+j-filter_size/2 >= X or y+i-filter_size/2 < 0 or y+i-filter_size/2 >= Y):
						temp_sum = temp_sum 
					else:	
						temp_sum = temp_sum + I_gray[y+i-filter_size/2][x+j-filter_size/2] * gauss_filter[i][j];
						#temp_sum = temp_sum/normalize 
			I_gauss[y][x] = temp_sum/normalize;
	return I_gauss	



# In[3]:


I = cv2.imread('Q1.jpeg',1)
I_gray = cv2.imread('Q1.jpeg',0)
#cv2.imshow('Gray' , I_gray)
cv2.imwrite('Q1_gray.jpeg' , I_gray)
Y = numpy.shape(I_gray)[0]
X = numpy.shape(I_gray)[1]
print("Y=",Y , " X=" , X)


# In[9]:


#FOR SMALL IMAGES, we can take a different gaussian


# In[20]:


#if(min(Y,X) > )
size = 7
sigma = 3.0
I_gauss2 = gauss_filter(I_gray, size, sigma)
cv2.imwrite('Q1_gauss_'+str(size)+'_' + str(sigma) + '.jpeg' , I_gauss2)


# In[21]:


mean_gray = numpy.mean(I_gray)
mean_gauss = numpy.mean(I_gauss2)
print(mean_gray)
print(mean_gauss)


# In[22]:


t_min = 0.3*mean_gauss
print(t_min)
t_max = 0.66*mean_gauss
print(t_max)


# In[128]:


max_val = 50


# In[129]:


# I_edges1 = cv2.Canny(I_gauss1,10,max_val)
# cv2.imwrite('Q1_edges7.jpeg', I_edges1)


# In[130]:


# I_inv2 = 255 - I_gauss2
# cv2.imwrite('Q1_inv11.jpeg', I_inv2)


# In[131]:


# I_edges2 = cv2.Canny(I_inv2,10,max_val)
# cv2.imwrite('Q1_edges_inv11.jpeg', I_edges2)


# In[35]:


c_min = 45
c_max = 80
I_edges2 = cv2.Canny(I_gauss2,c_min,c_max)
cv2.imwrite('Q1_edges'+ str(size) +'_' + str(sigma)+ '_' + str(c_min)+ '_' + str(c_max) + 'exp.jpeg', I_edges2)


# In[24]:


# I_edges3 = cv2.Canny(I_gauss3,30,max_val*2)
# cv2.imwrite('Q1_edges15.jpeg', I_edges3)


# In[25]:


print("canny done")


# In[26]:


# I_lap = laplacian(I_gauss1)
# cv2.imwrite('Q1_lap_7.jpeg', I_lap)


# In[36]:


#I_edges = I_edges1

#point = numpy.zeros(2)
#threshold = 0.7
#the more the > 200, the more should be r_min, if we take a square of half side = r_min, then there exists only 
#8*r_min points, now this number should be > 200 to make the cut off, this will give r_min = 25
r_min = 25
r_max = min(Y,X)/4
print(r_max)


# In[37]:


hoff_mat1 = find_circle(I_edges2, r_min, r_max, X , Y);
max_value = numpy.amax(hoff_mat1)
print("max_value: " , max_value)
#the value of R=0 => 0+r_min



# In[38]:


#removing those points whose count is less than 3*r half of 2pi*r
for i in range(r_min,numpy.shape(hoff_mat1)[2]):
    x, y = numpy.where(hoff_mat1[:,:,i] < 3*i)
    hoff_mat1[x[0],y[0]] = 0


# In[43]:


threshold = max_value/2
posY, posX, posR = numpy.where(hoff_mat1 > threshold)
print(posR)
print("")
print(posX)
# for i in range(r_min,R,1):
#     if(numpy.amax(hoff_mat))
# y_c, x_c, r_c = numpy.where(hoff_mat > 3)
# print("y_c")
# print(y_c)
# print("x_c")
# print(x_c)
# print("r_c")
# print(r_c)


# In[44]:


I_marked = numpy.zeros([Y,X,3])
I_marked2 = numpy.zeros([Y,X,3])
for y in range(Y):
    for x in range(X):
        I_marked[y][x] = I_gray[y][x];
        I_marked2[y][x] = I_edges2[y][x];
            
for i in range(numpy.size(posR)):
    for theta in range(0,360,1):
        a = int(round(posX[i] + posR[i]*numpy.cos(theta*pi/180)))
        b = int(round(posY[i] + posR[i]*numpy.sin(theta*pi/180)))
        #t = 1
        #for j in range(t):
        if(a >= 0 and a < X and b >= 0 and b < Y):
            I_marked2[b][a][1] = 255
            I_marked2[b][a][0] = 0
            I_marked2[b][a][2] = 0

            I_marked[b][a][1] = 255
            I_marked[b][a][0] = 0
            I_marked[b][a][2] = 0            
cv2.imwrite('Q1_marked_'+str(size)+'_'+str(sigma)+'_'+str(c_min)+'_' + str(c_max)+ '_' + str(threshold) +'.jpeg', I_marked) 
#cv2.imwrite('Q1_marked_11,3_22.jpg', I_marked) 
#         if(hoff_mat[y][x][275] > 255):
#             hoff_mat[y][x][275] = 255;
#cv2.imwrite('hough_trans1.jpg',hoff_mat[:][:][275])

# I_marked = numpy.zeros([Y, X, 3])
# for y in range(Y):
# 	for x in range(X):
# 		I_marked[y][x] = I_gray[y][x]

# for i in range(numpy.size(y_c)):
# 	iterations = 2*(2**r_c[i])
# 	step = pi/(2**r_c[i])
# 	if(r_c[i] > 7):
# 		iterations = 360;
# 		step = 1/(2*pi)
# 	for j in range(0,step,2*pi):
# 			y = int(round(y_c[i] + r_c[i]*sin(j)))
# 			x = int(round(x_c[i] + r_c[i]*cos(j)))
# 			I_marked[y][x][2] = 255;
# cv2.imwrite('I_marked.jpeg', I_marked)			


# In[47]:





# In[ ]:




