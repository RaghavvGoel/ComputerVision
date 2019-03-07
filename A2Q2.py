#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cv2 
import copy


# In[4]:


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print(cv2.TERM_CRITERIA_EPS , cv2.TERM_CRITERIA_MAX_ITER)


# In[22]:


n = 12;
m = 12;
objp = np.zeros([n*m,3],dtype = np.float32)


# In[23]:


k = 0; j = 0;
for i in range(n*m):
    objp[i][0] = k;
    objp[i][1] = j;
    if(k >= n-1):
        k = 0;
        j += 1;
    else:
        k +=1;
print(objp, " " , type(objp))        


# In[24]:


objpts = []
imgpts = []


# In[25]:


noi = 15
ret_arr = np.zeros(noi)
for i in range(noi):
    I_gray = cv2.imread('Q2/Left'+str(i+1)+'.bmp',0)
    #I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
#     print(I_gray)
#     print(I)
#     print("ylko" + str(i))
    ret, corners = cv2.findChessboardCorners(I_gray, (n,m), None)
    ret_arr[i] = ret;
    print(i , " " , ret)
    if(ret == True):
        objpts.append(objp)
        
        corners2 = cv2.cornerSubPix(I_gray,corners, (11,11), (-1,-1), criteria)    
        imgpts.append(corners)
        
        #cv2.drawChessboardCorners(I_gray, (n,m), corners2, ret)
        #cv2.imshow('image' , I)
        #cv2.imwrite('Q2_img.bmp' , I)
#         cv.waitKey(500)& 0xFF
# cv.destroyAllWindows()        


# In[26]:


# print(np.shape(objpts) , type(objpts[0][0][0]))
# print(np.shape(imgpts) , type(imgpts[0][0][0][0]))
# objpts_new = copy.deepcopy(objpts)
# objpts_new = np.zeros([np.shape(objpts)[0],np.shape(objpts)[1],np.shape(objpts)[2]], dtype="float32")
# print(np.shape(objpts_new))
# for i in range(np.shape(objpts)[0]):
#     for j in range(np.shape(objpts)[1]):
#         for k in range(np.shape(objpts)[2]):
#                    objpts_new[i][j][k] = np.float32(objpts[i][j][k])
#                    #objpts_new[i][j][k] = objpts_new[i][j][k].astype('float32')
# #objpts = objpts[:,:,:].astype('float32')
# #imgpts = imgpts[:,:,:].astype('float32')
# print(np.shape(objpts) , type(objpts_new[0][0][0]))


# In[27]:


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, I_gray.shape[::-1], None, None)


# In[28]:


print(I_gray.shape[::-1])


# In[29]:


print("ret")
print(ret)
print("mtx")
print(mtx)
print("tvecs")
print(tvecs)


# In[30]:


print(dist)


# In[31]:


print(rvecs)


# In[35]:


# for i in range(noi):
#     if(ret_arr[i] == True):
error = np.zeros(len(objpts))
# print(error)
# mean_error = 0;
# print(len(objpts))
print(error)


# In[36]:


mean_error = 0;
for i in range(len(objpts)):
    imgpts2, _ = cv2.projectPoints(objpts[i], rvecs[i], tvecs[i], mtx, dist)
    #print(np.shape(imgpts2) , type(imgpts2[0][0][0]))
    #print(np.shape(imgpts[i]), type(imgpts[i][0][0][0]))
    imgpts2 = np.float32(imgpts2)
    error[i] = cv2.norm(imgpts[i], imgpts2, cv2.NORM_L2)/len(imgpts2)
    mean_error += error[i]

print( "total error: {}".format(mean_error/len(objpts)) )    


# In[38]:


print(error)
print(cv2.NORM_L2)


# In[ ]:





# In[ ]:




