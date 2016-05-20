import numpy
from pylab import *
import pickle
import chainer
import chainer.functions as F
from chainer import optimizers
import cv2





def clip(img):


	black_x = [x for x in range(img.shape[1]) if sum(img[:,x,:]) < img.shape[0]*255*img.shape[2]]
	black_y = [y for y in range(img.shape[0]) if sum(img[y,:,:]) < 255*img.shape[1]*img.shape[2]]


	if len(black_x) != 0:

		left = min(black_x)
		right = max(black_x)
		top = min(black_y)
		bottom = max(black_y)
	
		img = img[top:bottom+1,left:right+1,:]



	if img.shape[0] > img.shape[1]:
		img_base = numpy.ones((img.shape[0],img.shape[0],3),numpy.uint8)*255
		pad = int((img.shape[0]-img.shape[1])/2)
		img_base[:,pad:-pad-(pad < (img.shape[0]-img.shape[1])/2.0)*1,:] = img

	else:
		img_base = numpy.ones((img.shape[1],img.shape[1],3),numpy.uint8)*255
		pad = int((img.shape[1]-img.shape[0])/2)
		if pad == 0:
			img_base = img
		else:
			img_base[pad:-pad-(pad < (img.shape[1]-img.shape[0])/2.0)*1,:,:] = img



	return img_base





def forward(x):

	h1 = F.max_pooling_2d(F.relu(model.conv1(x)),2)
	h2 = F.relu(model.conv2(h1))
	h3 = F.max_pooling_2d(F.relu(model.conv3(h2)),2)
	h4 = F.relu(model.layer1(h3))
	y = model.layer2(h4)

	return y








def main():


	T = 20000


	fp = open("./train_images","rb")

	MagicNumber = sum([ord(fp.read(1)) * pow(256,3-i) for i in range(4)])
	numImages = sum([ord(fp.read(1)) * pow(256,3-i) for i in range(4)])
	numRows = sum([ord(fp.read(1)) * pow(256,3-i) for i in range(4)])
	numCols = sum([ord(fp.read(1)) * pow(256,3-i) for i in range(4)])

	Imgs = []


	for i in range(T):
		print i
		img = numpy.zeros((numCols,numRows,3)).astype(numpy.float32)
		for y in range(numCols):
			for x in range(numRows):
				img[y,x,:] = (ord(fp.read(1)) < 64) * 255
		Imgs.append(img)

	fp.close()





	fp = open("./train_labels","rb")

	MagicNumber = sum([ord(fp.read(1)) * pow(256,3-i) for i in range(4)])
	numItems = sum([ord(fp.read(1)) * pow(256,3-i) for i in range(4)])



	Labels = zeros((T,1)).astype(numpy.int32)

	for i in range(T):
		Labels[i] = ord(fp.read(1))


	fp.close()





	for i in range(T):

		Imgs[i] = clip(Imgs[i])
		Imgs[i] = cv2.cvtColor(Imgs[i], cv2.COLOR_BGR2GRAY)
		Imgs[i] = cv2.resize(Imgs[i],(28,28))
		Imgs[i] = (Imgs[i] < 200) * 1
		Imgs[i] = Imgs[i].astype(numpy.float32)

#		imshow((Imgs[i]*100).astype(numpy.int32))
#		show()



####################################

	global model

	model = chainer.FunctionSet(
		conv1 = F.Convolution2D(1,32,5),
		conv2 = F.Convolution2D(32,64,4),
		conv3 = F.Convolution2D(64,128,3),
		layer1 = F.Linear(2048,1024),
		layer2 = F.Linear(1024,10)
	)


	optimizer = optimizers.SGD()
	optimizer.setup(model)



	for i in range(T):
		print i
		x = chainer.Variable(Imgs[i].reshape(1,1,28,28))
		t = chainer.Variable(Labels[i])
		y = forward(x)


		optimizer.zero_grads()
		loss = F.softmax_cross_entropy(y,t)
		loss.backward()
		optimizer.update()






	fp = open("./model.pickle","wb")
	pickle.dump(model,fp)
	fp.close()





main()


