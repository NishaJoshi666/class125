import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x,y = fetch_openml('mnist_786',version = 1, return_X_y = True)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 40,train_size = 7500,test_size = 2500)

xtrain_scaled = xtrain/255.0
xtest_scaled = xtest/255.0

lr = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(xtrain_scaled,ytrain)

def getPrediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bwresized = image_bw.resize((28,28),Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(image_bwresized,pixelfilter)
    invertedimage = np.clip(image_bwresize,pixelfilter)
    maxpixel = np.max(image_bwresized)
    invertedimagescaled = np.asarray(invertedimage/maxpixel)
    testsample = np.array(invertedimage).reshape(1,784)
    testpredict = lr.predict(testsample)
    return testpredict
