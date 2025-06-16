import pickle

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model=load_model(r'D:\Sabari\Sign_Language_Prediction\sign-language-detector-python-master (1)\sign-language-detector-python-master\V1\Integration/model.h5')
model_num=load_model(r'D:\Sabari\Sign_Language_Prediction\sign-language-detector-python-master (1)\sign-language-detector-python-master\V1\Integration/number_model.h5')
option_model=load_model(r'D:\Sabari\Sign_Language_Prediction\sign-language-detector-python-master (1)\sign-language-detector-python-master\V1\Integration/options_model.h5')
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'D',4:'E',5:'F',6:'G',
               7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
               14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
               20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

num_dict = {0: '0', 1: '1', 2: '2',3:'3',4:'4',5:'5',6:'6',
               7:'7',8:'8',9:'9'}

option_dict={0:'Delete',1:'Space'}

unique_characters = set()
list_final=[]

while True:
    key = cv2.waitKey(1) & 0xFF
    predicted_character=""
    predicted_option_character=""
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if len(data_aux)==42:
            data_aux.extend([0] * 42)
        else:
            pass

        if key == ord('a'):
            prediction = model.predict([np.asarray(data_aux).reshape(1, 84)])
            if np.max(prediction)>0.99:
                predicted_class_index = np.argmax(prediction[0])
                predicted_character = labels_dict[predicted_class_index]
        if key == ord('n'):
            prediction = model_num.predict([np.asarray(data_aux).reshape(1, 84)])
            if np.max(prediction)>0.99:
                predicted_class_index = np.argmax(prediction[0])
                predicted_character = num_dict[predicted_class_index]
        if key == ord('o'):
            prediction = option_model.predict([np.asarray(data_aux).reshape(1, 84)])
            if np.max(prediction)>0.99:
                predicted_class_index = np.argmax(prediction[0])
                predicted_option_character = option_dict[predicted_class_index]
        # Get the index of the class with the highest probability
        
        #print(prediction)
        

            #print(model.predict_proba([np.asarray(data_aux).reshape(1, 84)]))
            # Use the labels dictionary to get the corresponding character
           

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3,
                    cv2.LINE_AA)

    

    unique_characters.add(predicted_character)
    unique_val=''.join(unique_characters)    
    list_final.extend(unique_val)
    unique_text = ''.join(list_final)
    unique_characters=set()

    if predicted_option_character == 'Delete':
        print(list_final)
        
        if len(list_final)>0:
            list_final.pop()
        unique_text = ''.join(list_final)

    if predicted_option_character == 'Space':
        try:
            list_final.append(' ')
            print('Space Added!!')
        except:
            pass
    cv2.putText(frame, unique_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        
    if key == ord('q'):
        break

    # Display the predicted character if flag is set
    
        
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
