import cv2
import time
import sys
import numpy as np

def get_frame():
	# Helper Methods
	def buildGauss(frame, levels):     #image pyramid of different resolution downward
		pyramid = [frame]
		for level in range(levels):
			frame = cv2.pyrDown(frame)
			pyramid.append(frame)
		return pyramid
	def reconstructFrame(pyramid, index, levels): #image pyramid of different resolution upward
		filteredFrame = pyramid[index]
		for level in range(levels):
			filteredFrame = cv2.pyrUp(filteredFrame)
		filteredFrame = filteredFrame[:videoHeight, :videoWidth]
		return filteredFrame

	# Webcam Parameters
	webcam = None
	if len(sys.argv) == 2:
		webcam = cv2.VideoCapture(sys.argv[1])
	else:
		webcam = cv2.VideoCapture(0)
	realWidth = 500
	realHeight = 500
	videoWidth = 250
	videoHeight = 250
	videoChannels = 3
	videoFrameRate = 15
	webcam.set(3, realWidth)
	webcam.set(4, realHeight)


	#Color Magnification Parameters
	levels = 3
	alpha = 170
	minFrequency = 1.0
	maxFrequency = 2.0
	bufferSize = 150
	bufferIndex = 0

	# Output Display Parameters
	font = cv2.FONT_HERSHEY_SIMPLEX
	loadingTextLocation = (20, 80)
	bpmTextLocation = (videoWidth//2 + 5, 80)
	fontScale = 1
	fontColor = (255,255,255)
	lineType = 2
	boxColor = (0, 255, 0)
	boxWeight = 3

	# Initialize Gaussian Pyramid
	firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
	firstGauss = buildGauss(firstFrame, levels+1)[levels]
	videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
	fourierTransformAvg = np.zeros((bufferSize))

	# Bandpass Filter for Specified Frequencies
	frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
	#print(frequencies)     #0.0 to 14.9
	mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)  # band pass filter (1Hz to 2Hz as HR = 72bpm)

	# Heart Rate Calculation Variables
	bpmCalculationFrequency = 15    #Framerate=15
	bpmBufferIndex = 0
	bpmBufferSize = 10
	bpmBuffer = np.zeros((bpmBufferSize))

	i = 0
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	while True:
		ret, frame = webcam.read()
		if ret == False:
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,
	
			scaleFactor=1.2,
			minNeighbors=5
			,     
	   		minSize=(20, 20)
		)

		totalFace = len(faces)		

		detectionFrame = frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2), :]

		# Construct Gaussian Pyramid
		videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
		fourierTransform = np.fft.fft(videoGauss, axis=0)

		# Bandpass Filter
		fourierTransform[mask == False] = 0

		# Grab a Pulse
		if bufferIndex % bpmCalculationFrequency == 0:     #after every 15 frames
			i = i + 1
			for buf in range(bufferSize):
				fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
			hz = frequencies[np.argmax(fourierTransformAvg)]
			bpm = 60.0 * hz
			bpmBuffer[bpmBufferIndex] = bpm
			bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

		# Amplify
		filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
		filtered = filtered * alpha

		# Reconstruct Resulting Frame
		filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
		outputFrame = detectionFrame + filteredFrame
		outputFrame = cv2.convertScaleAbs(outputFrame)

		bufferIndex = (bufferIndex + 1) % bufferSize

		frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2), :] = outputFrame
		cv2.rectangle(frame, (int(videoWidth/2) , int(videoHeight/2)), (int(realWidth-videoWidth/2), int(realHeight-videoHeight/2)), boxColor, boxWeight)
		if(totalFace>0):
			if i > bpmBufferSize:
				cv2.putText(frame, "BPM: %d" % bpmBuffer.mean(), bpmTextLocation, font, fontScale, fontColor, lineType)
			else:
				cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)
		else:
			cv2.putText(frame, "Face Not Found..", loadingTextLocation, font, fontScale, fontColor, lineType)



		imgencode=cv2.imencode('.jpg',frame)[1]
		stringData=imgencode.tostring()
		yield (b'--frame\r\n'
			b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

	webcam.release()
	cv2.destroyAllWindows()