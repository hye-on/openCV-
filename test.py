import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

#%%
model = load_model('model.h5')
model.summary()

# open webcam (웹캠 열기)ㅃ
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    if not status:
        break

    img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    """
    # 이미지 색 바꾸기
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 잡음 제거
    img_hsv = cv2.fastNlMeansDenoisingColored(img_hsv,None,10,10,7,21)
    
    lower = np.array([0,48,80])
    upper = np.array([20,255,255])
    img_hand = cv2.inRange(img_hsv, lower, upper)
    
    # 경계선 찾음
    contours, hierachy = cv2.findContours(img_hand, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 영역 찾기
    max = 0
    maxcnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(max < area):
            max = area
            maxcnt = cnt
            
    # maxcontours 의 각 꼭지점 다각선 만들기
    hull = cv2.convexHull(maxcnt)
    
    # img 다 0으로 만들기?
    mask = np.zeros(img.shape).astype(img.dtype)
    color = [255, 255, 255]
    # 경계선 내부 255로 채우기
    cv2.fillPoly(mask, [maxcnt], color)
    img_hand = cv2.bitwise_and(img, mask)
    cv2.drawContours(img_hand, [maxcnt], 0, (255, 0, 0), 3)
    cv2.drawContours(img_hand, [hull], 0, (0, 255, 0), 3)
    """
    x = img_to_array(img)
    #x = img_to_array(img_hand)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prediction = model.predict(x)
    predicted_class = np.argmax(prediction[0])  # 예측된 클래스 0, 1, 2
    print(prediction[0])
    print(predicted_class)

    if predicted_class == 0:
        me = "바위"
    elif predicted_class == 1:
        me = "보"
    elif predicted_class == 2:
        me = "가위"

    # display
    fontpath = "font/AppleGothic"
    font1 = ImageFont.truetype(fontpath, 100)
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    draw.text((50, 50), me, font=font1, fill=(0, 0, 255, 3))
    frame = np.array(frame_pil)
    cv2.imshow('RPS', frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
