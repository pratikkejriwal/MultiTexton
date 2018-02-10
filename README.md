# Image Retrieval using Multi-Texton Histogram
Implementation of Content Based Image Retrieval Process based on Multi-Texton Histogram described by Guang-Hai Liu et al. in the [paper](https://www.sciencedirect.com/science/article/pii/S0031320310000907) using Python

## Multi-Texton Histogram
MTH is a generalized visual attribute descriptor but without any image segmentation or model training and is based on Julesz’s textons theory.
It can be used as both shape as well as color descriptor. The algorithm analyzes the spatial correlation between neighboring color and edge orientation based on four special texton types (depicted below), and then creates the texton co-occurrence matrix to describe the attributes using histogram.
#### Algorithm
- Image is split into Red, Blue, Green color channels
- Sobel Operator is applied to each of the channels (RGB color channel)
- Color Quantization in RGB color space
- Texton Detection: the figure below describes the textons that are to be detected
![Texton](https://i.imgur.com/HUCckpk.jpg)
#### Texton Detection Process
![texton detection](https://i.imgur.com/mjNH46X.jpg)

#### Repository Structure
- 226.jpg and 2712.jpg: Images from Corel-1k dataset used for visualization of the histogram in mth_retrieval.ipynb and Multi_Texton.ipynb respectively
- Multi_Texton.ipynb: Jupyter Notebook used for explaining code for extracting features using multi-texton of an input image
- mth_retrieval.ipynb: Jupyter Notebook used for explaining code for retrieval of image from the MongoDB database.
- MTH.py : Python code responsible for extracting the features from the images and seeding it into MongoDB database.
- retrieval.py : Python code responsible for retrieving the similar images from the database.
- dump/MTH: Folder containing the actual dump (82 bin feature-vector for each image) of the seeded images in the database. It can be restored as:
```
cd Directory
mongorestore --db db_name .
```

### Dataset used
Dataset used for the project is Corel-1k dataset which contains 1000 images from diverse contents such as building, horses, people, elephants, mountains, etc. Each image is of size 192×128 or 128×192 in the JPEG format.  The dataset can be downloaded from [Corel-1K](http://www.ci.gxnu.edu.cn/cbir/Dataset.aspx)
