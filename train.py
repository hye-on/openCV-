#from dataclasses import replace
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

path_dir1 = './rock/'
path_dir2 = './paper/'
path_dir3 = './scissors/' 

# path 에 존재하는 파일 목록 가져오기
file_list1 = os.listdir(path_dir1)
file_list2 = os.listdir(path_dir2)
file_list3 = os.listdir(path_dir3)

# train 용 이미지 준비
num = 0
train_img = np.float32(np.zeros((1468, 224, 224, 3))) # 198 + 522 + 747
train_label = np.float64(np.zeros((1468, 1)))

for img_name in file_list1:
    img_path = path_dir1+img_name
    img = load_img(img_path, target_size=(224,224))
    
    """
    # 이미지 색 바꾸기
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 잡음 제거
    img_hsv = cv2.fastNlMeansDenoisingColored(img_hsv, None, 10, 10, 7, 21)

    lower = np.array([0, 48, 80])
    upper = np.array([20, 255, 255])
    img_hand = cv2.inRange(img_hsv, lower, upper)

    # 경계선 찾음
    contours, hierachy = cv2.findContours(
        img_hand, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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
    x = img_to_array(img)  # x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num, :, :, :] = x
    
    train_label[num] = 0 # rock
    num = num+1
    
for img_name in file_list2:
    img_path = path_dir2+img_name
    img = load_img(img_path, target_size=(224,224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    train_img[num, :,:,:] = x
    
    train_label[num] = 1 # paper
    num = num+1
    
for img_name in file_list3 :
    img_path = path_dir3+img_name
    img = load_img(img_path, target_size=(224,224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    train_img[num,:,:,:] = x
    
    train_label[num] = 2 # scissors
    num = num+1
    
# 이미지 섞기
n_elem = train_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)

train_label = train_label[indices]
train_img = train_img[indices]

# cread the base pre-trained model
IMG_SHAPE = (224,224,3)

base_model = ResNet50(input_shape=IMG_SHAPE, weights = 'imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model : ", len(base_model.layers))

GAP_layer = GlobalAveragePooling2D()
dense_layer = Dense(3, activation=tf.nn.softmax) # nn. 다음에 선택지 더 있음.

model = Sequential([base_model,GAP_layer,dense_layer])

base_learning_rate = 0.001 # 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

model.fit(train_img, train_label, epochs=5)

#save model
model.save("model.h5")

print("Saved model to disk")