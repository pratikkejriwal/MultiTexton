# coding: utf-8

# # Image Retrieval based on Multi-Texton Histogram



#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import operator
from IPython.display import Image
import math


from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
collection = client.test_database.coral

db = []
num = np.int64(input('Enter the input Image: '))
for x in collection.find():
    db = np.array(x['distances'])

#Image(str(num)+'.jpg')

inputImage = db[num]

distance = np.zeros(1000*82).reshape(1000,82)

distance = abs(db - inputImage)/(1 + db + inputImage)

distanceSum = np.sum(distance,axis=1)

# In[19]:

keys = np.arange(len(distanceSum),dtype=int)

Imagedictionary = dict(zip(keys, distanceSum))
sorted_images = sorted(Imagedictionary.items(), key=operator.itemgetter(1))

# In[22]:

i = 0;
#Resultimages = []
#ResultHists = np.zeros(1*82).reshape(1,82)
for key in sorted_images:
    if(i<=20):
        #ResultHists[i]=db[key[0]]
        i = i +1
        imageName = str(key[0])+'.jpg'
        #Resultimages.append(imageName)
        print (imageName)
        cv2.imshow(str(i), cv2.imread(imageName))
    else:
        break;

cv2.waitKey(0)

# In[23]:


#for ima in Resultimages:
 #   imageD = Image(ima)
  #  display(imageD)

