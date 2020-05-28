#!/usr/bin/env python
# coding: utf-8

# # Intensity-level (monkey) entropy

# In[1]:


from PIL import Image, ImageFilter, ImageOps
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


# Given a discrete random variable $X$ on a probability space $(\Omega, \mathcal{F}, P)$ with image $\chi = \mathrm{im}(X)$, the *Shannon entropy* is
# $$
# H = \sum_{x \in \chi} -P(x) \ln P(x).
# $$
# The *intensity-level entropy* is the Shannon entropy of the empirical distribution of intensity values.

# In[2]:


def shannon_entropy(h):
    """The Shannon entropy in bits"""
    return -sum(p*np.log2(p) if p > 0 else 0 for p in h)

def intensity_entropy(data):
    hist, _ = np.histogram(data, bins=range(256+1), density=True)
    return shannon_entropy(hist)


# In[3]:


def uniform(n):
    return np.array([1] * n) / n

def rescale(data):
    b = np.max(data)
    a = np.min(data)
    return (b - data) / (b - a)


# ## Natural image

# In[4]:


img = ImageOps.grayscale(Image.open('test.jpg'))
scale = max(np.shape(img))
data = np.array(img)
img


# In[5]:


intensity_entropy(img)


# The problem with the intensity entropy is that it is usually near maximum (8 bits for these grayscale images).

# In[6]:


def intensity_blur(img, scales, display=True):
    scale = max(np.shape(img))
    
    results = []
    for k in scales:
        simg = img.filter(ImageFilter.GaussianBlur(k * scale))
        data = np.array(simg)
        ihist, ibins = np.histogram(data, bins=range(256+1), density=True)
        S = shannon_entropy(ihist)
        if display:
            hist = plt.hist(ibins[:-1], ibins, weights=ihist, alpha=0.5)
            results.append((k, simg, hist, S))
        else:
            results.append((k, S))
            
    if display:
        plt.axvline(x=np.mean(np.array(img)))
        
    return results


# In[7]:


results = intensity_blur(img, np.linspace(0, 1.5, num=50), False)

plt.plot(*np.transpose(results), 'o-')
plt.ylim((0, 8))
plt.xlabel = "Smoothing"
plt.ylabel = "Intensity Entropy (bits)"


# In[8]:


rimgs = [img for _, img, _, _ in intensity_blur(img, [0, 0.01, 0.05, 0.125, 0.25, 0.5])]
plt.show()


# In[9]:


_, axarr = plt.subplots(1, len(rimgs))
for i, subimg in enumerate(rimgs):
    axarr[i].imshow(subimg, cmap='gray')
plt.show()


# ## Random pixel values

# In[10]:


rsize = 250
randimg = Image.fromarray((256*np.random.rand(*2*[rsize])).astype('uint8'))
randimg


# ### Beware: GIGO
# The boundary effects and discrete kernel of `ImageFilter.GaussianBlur` renders the data unreliable after the "minimum" of the intensity entropy with smoothing. This is immediately clear after even small smoothing for random pixel values, since there are no spatial correlations.

# In[11]:


results = intensity_blur(randimg, np.linspace(0, 0.3, num=75), False)

plt.plot(*np.transpose(results), 'o-')
plt.ylim((0, 8))
plt.xlabel = "Smoothing"
plt.ylabel = "Intensity Entropy (bits)"


# In[12]:


rimgs = [img for _, img, _, _ in intensity_blur(randimg, [0.01, 0.05, 0.25])]


# In[13]:


plt.show()


# In[14]:


_, axarr = plt.subplots(1, len(rimgs))
for i, subimg in enumerate(rimgs):
    axarr[i].imshow(subimg, cmap='gray')
plt.show()


# The rightmost image should be uniform: the renormalization emphasizes incorrect deviations. These are what keep the intensity entropy from vanishing.

# ## Comparing different levels of smoothing

# Is composing $n$ Gaussian blurs with variance $\sigma^2$ the same as doing one with variance $n\sigma^2$ (considering the boundary effects and discrete kernel)?

# In[15]:


nsmooths = 10
cimg = img
oneimg = cimg.filter(ImageFilter.GaussianBlur(np.sqrt(nsmooths)*2))
oneimg


# In[16]:


nimg = cimg
for _ in range(nsmooths):
    nimg = nimg.filter(ImageFilter.GaussianBlur(2))
nimg


# Answer: **No**

# The differences between results at different scales can be pretty wack.

# In[17]:


Image.fromarray((255*rescale(np.array(nimg) - np.array(oneimg))).astype('uint8'))


# In[18]:


smimg = img
smdiff = np.array(smimg.filter(ImageFilter.GaussianBlur(2))) - np.array(smimg.filter(ImageFilter.GaussianBlur(100)))
diffimg = Image.fromarray((255 * rescale(smdiff)).astype('uint8'))
diffimg

