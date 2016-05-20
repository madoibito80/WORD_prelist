import glob
import cv2
import numpy
import time
import pickle
import chainer
import chainer.functions as F
from pylab import *






def forward(x):

	h1 = F.max_pooling_2d(F.relu(model.conv1(x)),2)
	h2 = F.relu(model.conv2(h1))
	h3 = F.max_pooling_2d(F.relu(model.conv3(h2)),2)
	h4 = F.relu(model.layer1(h3))
	y = model.layer2(h4)

	return y




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




def main():


	result = 0.0


	fp = open("./model.pickle","rb")
	global model
	model = pickle.load(fp)
	fp.close()


	files = glob.glob("./in/*")

	for file in files:

		print file

		img = cv2.imread(file)


		img_e = img
		w = 400.0 / img_e.shape[1]
		img_e = cv2.resize(img_e, (int(w*img_e.shape[1]), int(w*img_e.shape[0])))
		cv2.imshow("",img_e)



		ret, img = cv2.threshold(img,80,255,cv2.THRESH_BINARY)



		neiborhood8 = numpy.ones((2,2),numpy.uint8)
 		img= cv2.dilate(img,neiborhood8,iterations=1)
		neiborhood8 = numpy.ones((10,10),numpy.uint8)
		img= cv2.erode(img,neiborhood8,iterations=1)


		img = img[400:-400,:,:]


		black_x = [x for x in range(img.shape[1]) if sum(img[:,x,:]) < img.shape[0]*255*img.shape[2]*0.9]
		left = min(black_x)
		right = max(black_x)

		img = img[500:,left:right,:]



		black_y = [y for y in range(img.shape[0]) if sum(img[y,:,:]) < 255*img.shape[1]*img.shape[2]*0.4]
		top = min(black_y)
		bottom = max(black_y)

		top += 900
		bottom += 900

		img_clip = cv2.imread(file)
		img_clip = img_clip[top:bottom,left:right,:]

		img_clip = img_clip[10:-10,10:-10,:]

		print img_clip.shape

		w = 700.0 / img_clip.shape[1]
		img_clip = cv2.resize(img_clip, (int(w*img_clip.shape[1]), int(w*img_clip.shape[0])))


		ret,img_clip = cv2.threshold(img_clip,215,255,cv2.THRESH_BINARY)


		
###################################################



		img = cv2.imread(file)
		img = img[bottom:,left:right+1,:]


		w = 500.0 / img.shape[1]
		img = cv2.resize(img, (int(w*img.shape[1]), int(w*img.shape[0])))


		img = img[20:-60,:,:]



		black_x = [x for x in range(img.shape[1]) if sum(img[:,x,:]) < img.shape[0]*255*img.shape[2]*0.9]
		left = min(black_x)
		right = max(black_x)



		black_y = [y for y in range(img.shape[0]) if sum(img[y,:,:]) < 255*img.shape[1]*img.shape[2]*0.9]
		top = min(black_y)
		bottom = max(black_y)



		img = img[:bottom,:,:]


		num = [img[-187:-140,31:75,:]]

		num.append(img[-55:-6,32:76,:])

		num.append(img[-55:-6,125:165,:])

		num.append(img[-55:-6,213:258,:])

		num.append(img[-55:-6,305:350,:])


########################################################





		for i in range(len(num)):
			ret,num[i] = cv2.threshold(num[i],205,255,cv2.THRESH_BINARY)
			neiborhood8 = numpy.ones((3,3),numpy.uint8)
			num[i] = cv2.erode(num[i], neiborhood8,iterations=1)

			num[i] = clip(num[i])
		

		rec = ""


		for i in range(len(num)):


			num_img = cv2.resize(num[i],(28,28))
			num_img = cv2.cvtColor(num_img, cv2.COLOR_BGR2GRAY)
			num_img = (num_img < 180) * 1
			num_img = (num_img.copy().astype(numpy.float32))
#			imshow((num_img*100).astype(numpy.int32))
#			show()


			x = chainer.Variable(num_img.reshape(1,1,28,28))
			y = forward(x)
			y = numpy.argmax(y.data[0])
			
			rec += str(y)


		print rec



		print "please"
		num = raw_input()
		cv2.imwrite("./out/"+num+".png",img_clip)




		d = sum([0.2 for i in range(5) if num[i] == rec[i]])
		print d
		result += d



	print "[result]"
	print result / len(files)



main()














