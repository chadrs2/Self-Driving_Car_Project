###############################
#### Stop Car at Stop Sign ####
#### --------------------- ####
#### Objective:        ####
#### > Drive car forward   ####
####   until a sotpsign is ####
####   spotted on the cam  ####
###############################
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
from time import sleep
import cv2

# Inializations: #
#Camera fps/size
camera=PiCamera()
camera.resolution=(640,480)
camera.framerate=20
rawCapture=PiRGBArray(camera,size=(640,480))
sleep(0.1)
stopsign_cascade=cv2.CascadeClassifier('/home/pi/Desktop/oldRPi/RPi/stopsign_good.xml')
#Car PIN setups
enA=13
in1=27
in2=17
temp1=1
servo=12
angle=45
GPIO.setmode(GPIO.BCM)
GPIO.setup(enA,GPIO.OUT)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(servo,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
pwm_servo=GPIO.PWM(servo,100)
pwm_motor=GPIO.PWM(enA,1000)
pwm_servo.start(13)
pwm_motor.start(25)

# Begin Camera video and driving forward #
camera.start_preview()
camera.start_recording('stopsign_video.h264')
for frame in camera.capture_continuous(rawCapture,format="bgr",use_video_port=True):
    # Begin driving at medium speed
    pwm_motor.ChangeDutyCycle(35) #medium speed
    #pwm_motor.ChangeDutyCycle(80) #high speed
    #pwm_motor.ChangeDutyCycle(70) #med-hi speed
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    pwm_servo.ChangeDutyCycle(13)
    # grab raw NumPy array representing image - 3D array
    image=frame.array
    #convert image to grayscale
    gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Wait and read key input
    key=cv2.waitKey(1) & 0xFF
    # Find stopsign in image
    print("Before stop sign finding")
    found_stopsigns=stopsign_cascade.detectMultiScale(gray_img,1.1,5)
    print("Found "+str(len(found_stopsigns))+" stop sign(s)")
    if len(found_stopsigns)>0:
        for (x,y,w,h) in found_stopsigns:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
            camera.add_overlay(image)
            cv2.imwrite("found_stopsign_Jun_25_20.jpg",image)
            sign_width=w
            sign_height=h
            print("width of stop sign:",w,"and height:",h)
        if(sign_width>65 or sign_height>65):
            print("Turn on brake lights")
            print("Decrease motor speed")
            print("stop car")
            GPIO.output(in1,GPIO.LOW)
            GPIO.output(in2,GPIO.LOW)
            sleep(1)
            break
    #clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the 'q' key was pressed or a stop sign was found
    # break from the loop
    if key == ord("q"): #or len(found_stopsigns)>0:
            break
# End camera functions
camera.stop_recording()
camera.stop_preview()
cv2.destroyAllWindows()
camera.close()
# End car functions
pwm_servo.stop()
pwm_motor.stop()
GPIO.cleanup()
