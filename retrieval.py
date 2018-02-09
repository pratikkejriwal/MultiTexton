
# coding: utf-8

# # Image Retrieval based on Multi-Texton Histogram

# In[1]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import operator
from IPython.display import Image


# In[2]:


from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
collection = client.MTH.coralTest


# # taking input from the User

# In[3]:


db = []
path = "D:/SEM 6/Content/Lab_Activity/Corel 1K/image.orig/"
num = np.int64(input('Enter the input Image: '))
for x in collection.find():
    db = np.array(x['distances'])
Image(path + str(num)+'.jpg')


# In[7]:


inputImage = db[int(num)-1]

distance = np.zeros(1000*82).reshape(1000,82)
for i in range(1000):
    for j in range(82):
        distance[i,j] = abs(db[i,j] - inputImage[j])/(1 + db[i,j] + inputImage[j])

distanceSum = np.sum(distance,axis=1)


# In[8]:


keys = np.arange(len(distanceSum),dtype=int)

Imagedictionary = dict(zip(keys, distanceSum))
sorted_images = sorted(Imagedictionary.items(), key=operator.itemgetter(1))


# In[9]:


i = 0;
Resultimages = []
ResultHists = np.zeros(1000*82).reshape(1000,82)
for key in sorted_images:
    if(i<=20):
        print(key)
        ResultHists[i]=db[key[0]]
        i = i +1
        imageName = path + str(key[0])+'.jpg'
        Resultimages.append(imageName)
        print (imageName)
    else:
        break;


# In[10]:


for ima in Resultimages:
    imageD = Image(ima)
    display(imageD)

