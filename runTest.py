import numpy as np
import NN
from mlxtend.data import loadlocal_mnist


X_train, y_train = loadlocal_mnist(images_path='train-images.idx3-ubyte', 
                                    labels_path='train-labels.idx1-ubyte')


X_test, y_test = loadlocal_mnist(images_path='train-images.idx3-ubyte', 
                                labels_path='train-labels.idx1-ubyte')


output = np.array([ [1,0,0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,0,0,1]],dtype=float)

net = NN.Network([784,128,10])
for i, img in enumerate(X_train):
    img = img / 255.0 
    net.train(img,output[y_train[i]],epoch=20)
    print("Train percent:{}%".format(int(i/img.shape[0])),end='\r')

'''
import matplotlib.pyplot as plt
plt.imshow(X_test[0].reshape(28,28), cmap='gray')
plt.show()
'''
# it just tests 10 images, but for better performance you can implement a cost function with a higher epoch and whole test images files
for i in range(10):
    print("predict output:",net.predict(X_test[i]/255.0),"<-> real:",y_test[i])

