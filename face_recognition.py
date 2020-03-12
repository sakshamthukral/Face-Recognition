import cv2
import numpy as np
import os

############## KNN CODE ###############
def distance(v1,v2):
	#Euclidean
	return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
	dist = []

	for i in range(train.shape[0]):
		# Get the vector and label`
		# we have used the last column for the labels part 
		ix = train[i,:-1]
		iy = train[i,-1] # -1 because we have used the last column for the labels part 
		# Compute the distance from the test point
		d = distance(test,ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist,key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]

	# Get frequencies of each label
	output = np.unique(labels,return_counts=True)
	# Find max frequencyand corresponding label
	index = np.argmax(output[1])
	return output[0][index]
############################################

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("D:/Sublime/FaceRecognitionProject/haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './FaceData/'

face_data = []  # will contain the X-values ie feature information of data.X-values don't mean the x-coordinate values at all
labels = [] # contains the Y-values ie the indices of the data for algo

class_id = 0 # Labels for the given file
names = {}  # this dictionary Will be used to create a mapping between id and name77.9

# Data Preperation
for fx in os.listdir(dataset_path): # iterate over all the directories of the file
	if fx.endswith('.npy'):
		# Create a mapping btw class_id and name of the file
		names[class_id] = fx[:-4]
		print("Loaded : "+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		# create labels for the class,the following section of code of three lines ensures that each face in a single data_item or class,contains the same labels for ex., suppose saksham is the first class having class_id=0 , then each face in class saksham will have a label as 0.Similarly each face in class1 let it be of rahul.npy will contain the same label as 1 for each face in it.
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0) # axis=0 represents appending or concatenating in row wise,concatenating rows
face_labels = np.concatenate(labels,axis=0).reshape((-1,1)) # whole of this Y-data or the labels are now transformed into a single column

print(face_dataset.shape)
print(face_labels.shape)

# We now need to concatenate both X ie no. rows or datasets and Y ie indexes  in a single training matrix.
# We do this because our KNN algorithm excepts one training matrix in which we should have the X-data and the Y-data combined in single matrix
# Therefore for doing this we first need to reshape our Y-data int a single column as done above in line 60

trainset = np.concatenate((face_dataset,face_labels),axis=1) # axis=1 represents concatenating along the columns
print(trainset.shape)

# Testing

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face

		# Get the face Region Of Interest(ROI)
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# Predicted Label (out)
		out = knn(trainset,face_section.flatten())

		# Display on the screen the name and rectangle around it
		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()





