#################################################
### Controlling servo and motor from computer ###
#################################################
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
import cv2
import threading
import csv
import os

enB=13
in1=27
in2=17
temp1=1
servo=12
angle=45

GPIO.setmode(GPIO.BCM)
GPIO.setup(enB,GPIO.OUT)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(servo,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
pwm_servo=GPIO.PWM(servo,100)
pwm_motor=GPIO.PWM(enB,1000)
pwm_servo.start(12.75)
pwm_motor.start(25)

camera=PiCamera()
camera.resolution=(320,240)
camera.framerate=10
rawCapture=PiRGBArray(camera,size=(320,240))
sleep(0.1)

userinput = 'a'
servos_pwm = 12.75
motors_pwm = 25

file = open('data_img.csv','w',newline='')
writer = csv.writer(file)
img_idx = 1

print("\n")

def _print():
    print("Motor: '_'-stop g-go h-sprint s-slow")
    print("Servo: a-left q-minorleft w-straight d-right e-minorright")
    print("m-print menu again    z-exit")
def stop():
    print("stop")
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    pwm_motor.ChangeDutyCycle(25)
    global motors_pwm
    motors_pwm = 25
def straighten():
    pwm_servo.ChangeDutyCycle(12.75)
    global servos_pwm
    servos_pwm = 12.75
    sleep(.125)
def slow():
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    pwm_motor.ChangeDutyCycle(45)
    global motors_pwm
    motors_pwm = 45
def medium():
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    pwm_motor.ChangeDutyCycle(55)
    global motors_pwm
    motors_pwm = 55
def high():
    print("high")
    pwm_motor.ChangeDutyCycle(80)
    global motors_pwm
    motors_pwm = 80
def steer(angle):
    dc=float(angle)/10.0+2.5 #conversion from deg-DC
    pwm_servo.ChangeDutyCycle(dc+5)
    global servos_pwm
    servos_pwm = dc+5
    sleep(.125)

def beginrecording():
        for frame in camera.capture_continuous(rawCapture,format="rgb",use_video_port=True):
            if userinput is not 'z':
                image=frame.array
                #cv2.imshow("Frame",image)
                key=cv2.waitKey(1) & 0xFF
                rawCapture.truncate(0)
                global servos_pwm
                global motors_pwm
                global writer
                global img_idx
                directory = r'/home/pi/Desktop/AV/IMG'
                os.chdir(directory)
                curr_path = '~/Desktop/AV/IMG/'
                img_name = 'img_' + str(img_idx) + '.jpg'
                cv2.imwrite(img_name,image)
                writer.writerow([curr_path+img_name, servos_pwm, motors_pwm]) 
                #print("Servo PWM:",str(servos_pwm),"and motor PWM:",str(motors_pwm))
                img_idx = img_idx+1
            else:
                break

def drive():
    currangle=45
    _print()
    global userinput
    while(1):
        userinput=input()
        if userinput==' ':
            stop()
        elif userinput=='g':
            print("GO")
            medium()
        elif userinput=='h':
            print("SPRINT")
            high()
        elif userinput=='s':
            print("SLOW")
            slow()
        elif userinput=='w':
            currangle=45
            straighten()
        #Straight==12.5, Max left==8.5, & Max right=16.5
        elif userinput=='d': #steer right
            currangle=currangle+50
            steer(currangle)
        elif userinput=='a': #steer left
            currangle=currangle-27
            steer(currangle)
        elif userinput=='q': #steer minor left
            currangle=currangle-10
            steer(currangle)
        elif userinput=='e':#steer minor right
            currangle=currangle+30
            steer(currangle)
        elif userinput=='m':
            _print()
        elif userinput=='z':
            print("EXITING")
            break
        else:
            print("<<< wrong input >>>")
            print("please enter the defined data to continue...")

try:
    threading.Thread(target=drive).start()
    beginrecording()
except KeyboardInterrupt:
    pass

# Cleanup everything at end of file
cv2.destroyAllWindows()
camera.close()
file.close()
pwm_servo.stop()
pwm_motor.stop()
GPIO.cleanup()

