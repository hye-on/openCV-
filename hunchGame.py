import sys
import numpy as np
import cv2

from hunchGame_nam import TrackingROI, TrackingState


# 추적기능 상태
#얼굴 인식
TRACKING_STATE_CHECK = 0
#얼굴인식 위치를 기반으로 추적 기능 초기화
TRACKING_STATE_INIT = 1
#추적 동작
TRACKING_STATE_ON = 2

if __name__ == '__main__':
    #버전 출력
    print((cv2.__version__).split('.'))

    # 트레킹 함수 선택
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    # 기본 KCF(Kernelized Correlation Filters)가 속도가 빠르다.
    tracker_type = tracker_types[2]
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

# 비디오 파일 열기
#cap = cv2.VideoCapture('camshift.avi')

face_cascade = cv2.CascadeClassifier()
#얼굴인식용 haar 불러오기
face_cascade.load(
    './haarcascades/haarcascade_frontalface_default.xml')


# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)
 
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

#  if not cap.isOpened():
#      print('Video open failed!')
#      sys.exit()

# 초기 사각형 영역: (x, y, w, h)
x, y, w, h = 320, 110, 100, 100
TrackingROI = (x, y, w, h) # rc



ret, frame = webcam.read()

if not ret:
    print('frame read failed!')
    sys.exit()

"""
roi = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


# HS 히스토그램 계산
channels = [0, 1]
ranges = [0, 180, 0, 256]
hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)



# Mean Shift 알고리즘 종료 기준
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

"""
while webcam.isOpened():

# 비디오 매 프레임 처리
#while True:
    ret, frame = webcam.read() # ret = ok

    if not ret:
        break

    # 추적 상태가 얼굴 인식이면 얼굴 인식 기능 동작 
    # 처음에 무조건 여기부터 들어옴.
    if TrackingState == TRACKING_STATE_CHECK:
        # 흑백 변경 
        grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 히스토그램 평활화(재분할)
        grayframe = cv2.equalizeHist(grayframe)
        # 얼굴인식 
        faces = face_cascade.detectMultiScale(grayframe, 1.1, 5, 0, (30,30))
        
        # 얼굴이 하나라도 잡혔다면 
        if len(faces) > 0:
            x,y,w,h = faces[0]
            #인식된 위치 및 크기를 TrackingROI 에 저장.
            TrackingROI = (x,y,w,h) # TrackingROI
            #인식된 얼굴 표시 순식간에 지나가서 거의 볼 수 없음(녹색)
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),3,4,0)
            #추적 상태를 추적 초기화로 변경
            TrackingState = TRACKING_STATE_INIT
            print('det w : %d ' % w + 'h : %d ' % h)
            
    # 추적 초기화
    # 얼굴이 인식되면 동작함 
    elif TrackingState == TRACKING_STATE_INIT:
        #추적 함수 초기화 
        #얼굴 인식으로 가져온 위치와 크기를 함께 넣어준다.
        ret = tracker.init(frame, TrackingROI)
        if ret:
            # 성공하였다면 추적 동작상태로 변경
            TrackingState = TRACKING_STATE_ON
            print('tracking init succeeded')
        else:
            # 실패하였다면 얼굴 인식 상태로 다시 돌아감
            TrackingState = TRACKING_STATE_CHECK
            print('tracking init failed')
    # 추적동작 
    elif TrackingState == TRACKING_STATE_ON:
        #추적 
        ret, TrackingROI = tracker.update(frame)
        if ret:
            # 추적 성공했다면
            p1 = (int(TrackingROI[0]), int(TrackingROI[1]))
            p2 = (int(TrackingROI[0] + TrackingROI[2]), int(TrackingROI[1]+TrackingROI[3]))
            # 화면에 박스로 표시(파랑)
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            print('success x %d ' % (int(TrackingROI[0])) + 'y %d ' % (int(TrackingROI[1])) +
                  'w %d ' % (int(TrackingROI[2])) + 'h %d ' % (int(TrackingROI[3])))
        else:
            print('Tracking failed')
            
            TrackingState = TRACKING_STATE_CHECK
            
    """
    # HS 히스토그램에 대한 역투영
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)
    """
    
    """
    # Mean Shift
    _, rc = cv2.meanShift(backproj, rc, term_crit)

    
    #print("가나다라마바")
    if rc[1]==0:
        cv2.rectangle(frame, rc, (0, 225, 0), 2)
    
    # 추적 결과 화면 출력
    else:
        cv2.rectangle(frame, rc, (0, 0, 255), 2)
    
    """
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

webcam.release()
cv2.destroyAllWindows()  
