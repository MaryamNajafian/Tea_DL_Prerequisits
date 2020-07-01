"""
scipy:
    * PDF,
    * CDF,
    * Convolution, etc
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# from scipy.stats import gamma
# from scipy.stats import beta

x = np.linspace(-6, 6, 1000)

#%%
# probability desnsity function at x of the given
# for the normal distribution
# loc: is the mean
# scale: standard deviation

fx = norm.pdf(x, loc=0, scale=1)
plt.plot(x,fx)
plt.show()

fx = norm.cdf(x, loc=0, scale=1)
plt.plot(x,fx)
plt.show()

log_fx = norm.logpdf(x, loc=0, scale=1)
plt.plot(x,log_fx)
plt.show()

#%% convolution

print("plt image data sets")
import wget
from PIL import Image

print('Beginning file download with wget module')
url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
wget.download(url)
#%%
import matplotlib.pyplot as plt
im = Image.open('mqdefault.jpg')
#2D conv is made for 2D image so we don't want that 3rd dimension
# so we convert to grey scale to get rid of the 3rd Dimension
gray = np.mean(im, axis = 2)

x = np.linspace(-6, 6, 50)
x.shape # (50,)

fx = norm.pdf(x, loc=0, scale=1)
fx.shape # (50,)

guassian_filter = np.outer(fx,fx) # outer product of fx by itself
guassian_filter.shape #(50, 50)

plt.imshow(guassian_filter, cmap = 'gray')
plt.show()


#%%
from scipy.signal import convolve2d
out = convolve2d(guassian_filter,gray)
plt.subplot(1,2,1)
plt.imshow(guassian_filter, cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(out, cmap = 'gray')
plt.show()