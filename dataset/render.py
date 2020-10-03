#coding:utf-8
import numpy as np
import cv2


def genarateMotionBlurPsf(length,angle):
	EPS=np.finfo(float).eps
	alpha = (angle-np.floor(angle/ 180) *180) /180* np.pi
	cosalpha = np.cos(alpha)  
	sinalpha = np.sin(alpha)  
	if cosalpha < 0:
		xsign = -1
	elif angle == 90:
		xsign = 0
	else:  
		xsign = 1
	psfwdt = 1;  
	#模糊核大小
	sx = int(np.fabs(length*cosalpha + psfwdt*xsign - length*EPS))  
	sy = int(np.fabs(length*sinalpha + psfwdt - length*EPS))
	psf1=np.zeros((sy,sx))

	#psf1是左上角的权值较大，越往右下角权值越小的核。
	#这时运动像是从右下角到左上角移动
	for i in range(0,sy):
		for j in range(0,sx):
			psf1[i][j] = i*np.fabs(cosalpha) - j*sinalpha
			rad = np.sqrt(i*i + j*j) 
			if  rad >= length//2 and np.fabs(psf1[i][j]) <= psfwdt:  
				temp = length//2 - np.fabs((j + psf1[i][j] * sinalpha) / cosalpha)  
				psf1[i][j] = np.sqrt(psf1[i][j] * psf1[i][j] + temp*temp)
			psf1[i][j] = psfwdt + EPS - np.fabs(psf1[i][j]);  
			if psf1[i][j] < 0:
				psf1[i][j] = 0
	#运动方向是往左上运动，锚点在（0，0）
	anchor=(0,0)
	#运动方向是往右上角移动，锚点一个在右上角
	#同时，左右翻转核函数，使得越靠近锚点，权值越大
	if angle<90 and angle>0:
		psf1=np.fliplr(psf1)
		anchor=(psf1.shape[1]-1,0)
	elif angle>-90 and angle<0:#同理：往右下角移动
		psf1=np.flipud(psf1)
		psf1=np.fliplr(psf1)
		anchor=(psf1.shape[1]-1,psf1.shape[0]-1)
	elif angle<-90:#同理：往左下角移动
		psf1=np.flipud(psf1)
		anchor=(0,psf1.shape[0]-1)
	psf1=psf1/psf1.sum()
	return psf1,anchor

def adjustGamma(img, gamma=1.0, mask=None):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	img_out = cv2.LUT(img, table)
	if mask is not None:
		if img.ndim==3:
			mask = np.stack([mask]*3, axis=2)
		img_out = img_out*mask + img*(1-mask)
	return img_out.astype(np.uint8)


def jpegQuality(img, quality=90):
	result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
	decimg = cv2.imdecode(encimg, 1)
	return decimg

def adjustBrightAndContrast(img, bright=1.0, contrast=1.0, mask=None):
	img = img.astype(np.float32)
	imgAVG = np.mean(img, axis=(0,1))
	if bright<1:
		bright *= bright
	img = np.clip((img - imgAVG)*contrast + bright*imgAVG, 0, 255).astype(np.uint8)
	return img


	
def randomGuassianBlur(img, probability=0.5):
	if np.random.random() > probability:
		return img
	ksize = 3
	r2 = np.random.random()
	sigma = (0.3*((ksize-1)*0.5-1)+0.8)*r2
	img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
	return img

def randomMotionBlur(img, probability=0.5, distanceMax=5):
	r1 = np.random.random(2)
	angle = r1[0]*360.0
	distance = distanceMax*r1[1]
	kernel, anchor = genarateMotionBlurPsf(distance,angle)
	img=cv2.filter2D(img, -1, kernel, anchor=anchor)
	return img


def util_lightStainMask(h, w, maxSize, center):
	minSize = maxSize/5
	r1 = np.random.random(5)
	if center is None:
		center = (int(r1[0]*w), int(r1[1]*h))
	axes = (minSize+r1[2]*maxSize, minSize+r1[3]*maxSize)
	#axes = (maxSize, 0.2*maxSize)
	angle = r1[4]*np.pi
	angleD = int(r1[4]*180)
	mapX, mapY = np.meshgrid(np.arange(w), np.arange(h))
	points = np.vstack([mapX.reshape(-1)-center[0], mapY.reshape(-1)-center[1]])
	matRot22 = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
	matScale = np.array([[1/axes[0],0],[0,1/axes[1]]])
	#points = matRot22.dot(matScale).dot(points)
	points = matScale.dot(matRot22.dot(points))
	mask = np.exp(-np.power(points,2)/2)
	mask0 = mask[0].reshape(h,w)
	mask1 = mask[1].reshape(h,w)
	mask0/=np.max(mask0)
	mask1/=np.max(mask1)
	mask = mask[0]*mask[1]
	mask = mask.reshape(h,w)
	mask = mask / np.max(mask) * 3
	return mask

def randomLightStain(img, maxSize = 30, up=10, down=4, probability=0.5, center=None):
	if np.random.random() > probability:
		return img
	h,w = img.shape[:2]
	mask = np.zeros((h,w))
	for i in range(np.random.randint(3)+1): #1 or 2 or 3
		mask += util_lightStainMask(h, w, maxSize, center)
	mask[mask>1.0]=1.0
	r2 = np.random.random()*(up-down)+down # 4-10
	img2 = adjustGamma(img, gamma=r2, mask=mask)
	return img2
