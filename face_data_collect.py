# Write a Python Script that captures images from your webcam video stream
# Extract all Faces from the image frame (using harcascades )
# Store the face information into numpy arrays

# 1. Read and show video stream,capture images
# 2. Detect Faces and show bonding box(haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data


import cv2
import numpy as np

# init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("D:/Sublime/FaceRecognitionProject/haarcascade_frontalface_alt.xml")

skip = 0 # Counter that will be used to every 10th face
face_data = [] # array that will store every 10th face
dataset_path =  './FaceData/'# './data/' # that will store the array of daata in this path or folder
file_name =  "Anu" #input("Enter the name of the person : ") 

while True:
	ret,frame = cap.read()

	if ret == False:
		continue

	gray_frame = cv2.cvtColor(cv2.UMat(frame),cv2.COLOR_BGR2GRAY) # we stored the images in gray_frame to save the memory 

	
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key=lambda f:f[2]*f[3]) #add reverse=True as third parameter if you want the largest face in front of the list else do as done in code

	# Pick the last face because it is the largest face according to area(f[2]*f[3])
	
	# The line in which in you are initializing “face_section=[]” is causing the error.To correct this, just replace it with"face_section = np.ones((100,100))255.0"
    # This line just makes 100100 numpy array having 255.0 in each cell, actually 255.0 corresponds to white color on grayscale. And this needs to be done so that if no face_section is initialized than it contains empty list. And as you said only 10% of time it runs and fails in rest. That’s the reason.
    # So just change this And then your code would work fine.
	face_section = np.ones((100,100))*255.0
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

		# Extract one section of the face (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		# Store every 10th Face
		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))
	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1)) # number of rows should be same as number of faces ie face_data.shape[0], number of columns should be identified automatically thats why we gave as -1
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully Saved at"+dataset_path+file_name+'.npy')
  
cap.release()
cv2.destroyAllWindows()