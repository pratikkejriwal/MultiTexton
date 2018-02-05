# coding: utf-8

# # Image Retrieval Based on Multi-Texton Histogram



# importing libraries
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')

Database = np.zeros(1000*82).reshape(1000,82)
for entry in range(1000):
    imagename = str(entry)+'.jpg'
    img = cv2.imread(imagename)
    width, height, channels = img.shape
    
    # # Texture Orientation Detection
    CSA = 64
    CSB = 18
    arr = np.zeros(3*width*height).reshape(width,height,3)
    ori = np.zeros(width * height).reshape(width, height)
    gxx = gyy = gxy = 0.0
    rh = gh = bh = 0.0
    rv = gv = bv = 0.0
    theta = np.zeros(width*height).reshape(width,height)
    
    for i in range(1, width-1):
        for j in range(1, height-1):
            rh=arr[i-1,j+1,0] + 2*arr[i,j + 1,0] + arr[i+1, j+1,0] - (arr[i-1, j - 1, 0] + 2 * arr[i,j-1, 0] + arr[i + 1, j - 1, 0])
            gh=arr[i-1,j+1,1] + 2*arr[i,j + 1,1] + arr[i+ 1,j+1,1] - (arr[i-1, j - 1, 1] + 2 * arr[i,j-1, 1] + arr[i + 1, j - 1, 1])
            bh=arr[i-1,j+1,2] + 2*arr[i,j + 1,2] + arr[i+ 1,j+1,2] - (arr[i-1, j - 1, 2] + 2 * arr[i,j-1, 2] + arr[i + 1, j - 1, 2])
            rv=arr[i+1,j-1,0] + 2*arr[i+1, j, 0] + arr[i+ 1,j+1,0] - (arr[i-1, j - 1, 0] + 2 * arr[i-1,j, 0] + arr[i - 1, j + 1, 0])
            gv=arr[i+1,j-1,1] + 2*arr[i+1, j, 1] + arr[i+ 1,j+1,1] - (arr[i-1, j - 1, 1] + 2 * arr[i-1,j, 1] + arr[i - 1, j + 1, 1])
            bv=arr[i+1,j-1,2] + 2*arr[i+1, j, 2] + arr[i+ 1,j+1,2] - (arr[i-1, j - 1, 2] + 2 * arr[i-1,j, 2] + arr[i - 1, j + 1, 2])
            
            gxx = math.sqrt(rh * rh + gh * gh + bh * bh)
            gyy = math.sqrt(rv * rv + gv * gv + bv * bv)
            gxy = rh * rv + gh * gv + bh * bv
            
            theta[i,j] = (math.acos(gxy / (gxx * gyy + 0.0001))*180 / math.pi)
    
    ImageX = np.zeros(width * height).reshape(width, height)
    # Color Quantization in RGB Color Space
    R = G = B = 0
    VI = SI = HI = 0
    BC,GC,RC = cv2.split(img)    
    for i in range(0, width):
        for j in range(0, height):
            R = RC[i][j]
            G = GC[i][j]
            B = BC[i][j]
            
            if (R >=0 and R <= 64):
                VI = 0;
            if (R >= 65 and R <= 128):
                VI = 1;
            if (R >= 129 and R <= 192):
                VI = 2;
            if (R >= 193 and R <= 255):
                VI = 3;
            if (G>= 0 and G <= 64):
                SI = 0;
            if (G >= 65 and G <= 128):
                SI = 1;
            if (G >= 129 and G <= 192):
                SI = 2;
            if (G >= 193 and G <= 255):
                SI = 3;
            if (B >= 0 and B <= 64):
                HI = 0;
            if (B >= 65 and B <= 128):
                HI = 1;
            if (B >= 129 and B <= 192):
                HI = 2;
            if (B >= 193 and B <= 255):
                HI = 3;
            
            ImageX[i, j] = 16 * VI + 4 * SI + HI;
            
    for i in range(0, width):
        for j in range(0, height):
            ori[i,j] = round(theta[i,j]*CSB/180);
            
            if(ori[i,j]>=CSB-1):
                ori[i,j]=CSB-1;
    
    # # Texton Detection
    
    Texton = np.zeros(width * height).reshape(width, height)
    
    for i in range(0,(int)(width/2)):
        for j in range(0,(int)(height/2)):
            if(ImageX[2*i,2*j] == ImageX[2*i+1,2*j+1]):
                Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j];
                Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j];
                Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1];
                Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1];
            
            if (ImageX[2*i,2*j+1] == ImageX[2*i+1,2*j]):
                Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j];
                Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j];
                Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1];
                Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1];
                
            if (ImageX[2*i,2*j] == ImageX[2*i+1,2*j]): 
                Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j];
                Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j];
                Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1];
                Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1];
            
            if (ImageX[2*i,2*j] == ImageX[2*i,2*j+1]):
                Texton[2 * i, 2 * j] = ImageX[2 * i, 2 * j];
                Texton[2 * i + 1, 2 * j] = ImageX[2 * i + 1, 2 * j];
                Texton[2 * i, 2 * j + 1] = ImageX[2 * i, 2 * j + 1];
                Texton[2 * i + 1, 2 * j + 1] = ImageX[2 * i + 1, 2 * j + 1];                   
                
    # # Multi-Texton Histogram
    
    MatrixH = np.zeros(CSA + CSB).reshape(CSA + CSB)
    MatrixV = np.zeros(CSA + CSB).reshape(CSA + CSB)
    MatrixRD = np.zeros(CSA + CSB).reshape(CSA + CSB)
    MatrixLD = np.zeros(CSA + CSB).reshape(CSA + CSB)
    
    D = 1 #distance parameter
    
    for i in range(0, width):
        for j in range(0, height-D):
            if(ori[i, j+D] == ori[i, j]):
                MatrixH[(int)(Texton[i,j])] += 1;
            if(Texton[i, j + D] == Texton[i, j]):
                MatrixH[(int)(CSA + ori[i, j])] += 1;
    
    for i in range(0, width-D):
        for j in range(0, height):
            if(ori[i + D, j] == ori[i, j]):
                MatrixV[(int)(Texton[i,j])] += 1;
            if(Texton[i + D, j] == Texton[i, j]):
                MatrixV[(int)(CSA + ori[i, j])] += 1;
    
    for i in range(0, width-D):
        for j in range(0, height-D):
            if(ori[i + D, j + D] == ori[i, j]):
                MatrixRD[(int)(Texton[i,j])] += 1;
            if(Texton[i + D, j + D] == Texton[i, j]):
                MatrixRD[(int)(CSA + ori[i, j])] += 1;
    
    for i in range(D, width):
        for j in range(0, height-D):
            if(ori[i - D, j + D] == ori[i, j]):
                MatrixLD[(int)(Texton[i,j])] += 1;
            if(Texton[i - D, j + D] == Texton[i, j]):
                MatrixLD[(int)(CSA + ori[i, j])] += 1;
    
    # # Feature Vectors
    MTH = np.zeros(CSA + CSB).reshape(CSA + CSB)
    
    for i in range(0, CSA + CSB):
        MTH[i] = (MatrixH[i] + MatrixV[i] + MatrixRD[i] + MatrixLD[i])/4.0;
    MTH

    for i in range(CSA+CSB):
        Database[entry] = MTH[i]
    print("Entered for"+imagename)
    
collection = client.test_database.coral
collection.insert({"distances":Database.tolist(),"name":'Coral Dataset'})
print (Database[0])



# In[50]:


#plt.axis([0, 82, 0, 6000])
#plt.bar(np.arange(82),MTH)
#plt.xlabel('Bin size')
#plt.ylabel('Frequency')
#plt.title('Histogram of MTH')
#plt.grid(True)

#plt.show()
