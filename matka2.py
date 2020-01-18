import cv2
from mtcnn.mtcnn import MTCNN
import face_recognition
import numpy as np
from matplotlib import pyplot


detector = MTCNN()

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# # Load a second sample picture and learn how to recognize it.
teja_image = face_recognition.load_image_file("face.jpg")
teja_face_encoding = face_recognition.face_encodings(teja_image)[0]

# Load a third sample picture and learn how to recognize it.
my_image = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    teja_face_encoding,
    my_face_encoding
]
known_face_names = [
    "Barack Obama",
    "teja",
    "Mishra"
]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def draw_faces(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        # define subplot
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
        # plot face
        try:
            pyplot.imshow(data[y1:y2, x1:x2])
        except Exception:
            pass
    pyplot.show()


cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        i = 0
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
    

            x1, y1, width, height = bounding_box
            x2, y2 = x1 + width, y1 + height
            # define subplot
            # pyplot.subplot(1, len(result), i+1)
            # pyplot.axis('off')
            # # plot face
            dola_re = frame[y1:y2, x1:x2] 
            # try:
            #     pyplot.imshow(dola_re)
            # except Exception:
            #     pass  

            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(dola_re)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left, bottom - 35), font, 1.0, (255, 255, 255), 1)

            print("face_names:",face_names)



            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
            if face_names!=[]:
                cv2.putText(frame, face_names[0], (bounding_box[0] + 6, bounding_box[2] - 6), font, 1.0, (0,200,155), 1)                
    
            cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
            i+=1

            #pyplot.show()
 

            
    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()