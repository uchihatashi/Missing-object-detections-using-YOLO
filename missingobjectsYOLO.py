import cv2
import numpy as np

def detectobject(img):
    height, width, channels = img.shape

    # Using blob function of opencv to preprocess image
    #blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    #print(blob.shape)
    """blobb = blob.reshape(blob.shape[2],blob.shape[3],blob.shape[1])
    cv2.imshow('Blob',blobb)"""

    # Detecting objects 
    net.setInput(blob)
    outs = net.forward(output_layers)

    # checking information to show on the screen
    class_ids = []  
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    labels = ""
    #showing informations on the screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            confi = round(confidences[i], 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + str(confi), (x, y-7), font, 0.7, color, 1, lineType=cv2.FILLED)
            labels = labels + label + "," 
    
    return labels


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

# loading each class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#to get the random colors
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

old_label = ""
missinglabel = ""
cispress = 0

print("for new image press n  and to check the missing image press c or C", "\nPress Q or q to quit")
file2 = open(r"record.txt","w+") 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = frame

    try:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        labels = detectobject(img)
        #waiting for the key press
        k = cv2.waitKey(20)
        
        # press n for uploading a new image with the object
        if k & 0xFF == ord('n'):
            cispress = 0
            old_label = labels
            print(old_label)


        # press c for checking the object is missing or not.
        if k & 0xFF == ord('c'):
            cispress = 1
            missinglabel = ""
            
            labels_list = list(labels.split(','))
            old_label_list = list(old_label.split(','))
            print(labels_list)
            print(old_label_list)

            for label in old_label_list:
                if(label not in labels_list):
                    missinglabel += label + "," 
                    
            if missinglabel != "":
                missinglabel = "Missing objects: " + missinglabel
                print("Missing objects: " + missinglabel)

            else:
                missinglabel = "No missing object"                
                print("No missing object")

        if  k & 0xFF == ord('q'):
            break
        
        # it prints the object is missing or not on the cam frame
        if cispress == 1:
            cv2.putText(img, missinglabel, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,0,255], 2, lineType=cv2.FILLED)

        cv2.imshow("Image", img)

    except Exception as e:
        print(str(e))

cv2.destroyAllWindows()







#download link: https://www.kaggle.com/valentynsichkar/yolo-coco-data?select=coco.names
#               https://pjreddie.com/darknet/yolo/
# 





