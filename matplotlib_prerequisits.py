"""
MatPlotLib topics:
    Line Charts: any one dimensional signal (sound, time series)
    Scatter plot: usualy we use dimensionality reduction techniques to be able to visualize the geometric distribution of the data points
    Histogram: show probability distribution of data
    Plotting image datasets: helpful in computer vision applications
"""
#%%
print("Line Charts")
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,20,100)
y = np.sin(x) + 0.2 * x
plt.plot(x,y)

plt.xlabel('input')
plt.ylabel('output')
plt.title("my line chart plot")
plt.show()

#%%
print("Histogram")
import numpy as np
import matplotlib.pyplot as plt

# it shows randn samples from a standard normal distribution
x = np.random.randn(1000)
plt.hist(x,bins=50)
plt.show()

# it shows random samples from a uniform [0-1) distribution
y = np.random.random(1000)
plt.hist(y,bins=50)
plt.show()

#%% use wget to download image from internet
"""
* download image with WGET and load with PILLOW
    > pip install wget
    > pip install Pillow

* download all images from a website and
  it stores the original hierarchy of the site with all the 
  subfolders and so the images are dotted around.

* download all the images into a single folder:
  wget -r -A jpeg,jpg,bmp,gif,png http://www.somedomain.com

"""
#%%
print("plt image data sets")
import wget
from PIL import Image

print('Beginning file download with wget module')
url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
wget.download(url)

print('Beginning to load the image file and convert to numpy array')
image_file = Image.open('mqdefault.jpg')
image_file_type = type(image_file) # PIL.JpegImagePlugin.JpegImageFile
image_file_numpy = np.array(image_file)
image_file_numpy_shape = image_file_numpy.shape # (180, 320, 3)
#%%
print('Beginning image show for both original and numpy version of the image')

plt.imshow(image_file_numpy)
plt.show()

plt.imshow(image_file)
plt.show()
#%%
# conver color image to black and white image
# by taking a mean across the color channels

print("Beginning to convert color to black-white image")

image_file_numpy_grayscale = image_file_numpy.mean(axis=2)
image_file_numpy_grayscale.shape
# shows a heatmap  version of the pic
plt.imshow(image_file_numpy_grayscale)
plt.show()
# shows a grayscale version of the pic
plt.imshow(image_file_numpy_grayscale, cmap='gray')
plt.show()
#%%

