#######################################
### DRIVE CAR IN LANES AUTONOMOUSLY ###
#######################################
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
from time import sleep
import cv2
import os
import tflite_runtime.interpreter as tflite

#Load tflite model and allocate tensors
interpreter = tflite.Interpreter(model_path='/home/pi/Desktop/AV/model.tflite')
interpreter.allocate_tensors()
#Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
#floating_model = (input_details[0]['dtype'] == np.float32)
#interpreter.resize_tensor_input(input_details[0]['index'], (32, 320, 240, 3))
#interpreter.resize_tensor_input(output_details[0]['index'], (32, 5))
#interpreter.allocate_tensors()
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])

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
pwm_servo.start(13)#(12.5)
pwm_motor.start(25)

camera=PiCamera()
camera.resolution=(320,240)
camera.framerate=5
rawCapture=PiRGBArray(camera,size=(320,240))
sleep(0.1)

servos_pwm = 13
motors_pwm = 25

def img_preprocess(img):
  #img = mpimg.imread(img_path)
  img = img[20:240, :, :]
  #YUV is important when using NVIDIA Model
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) #Y=luminescence; U,V=chrominance
  img = cv2.GaussianBlur(img, (3,3), 0) #helps get rid of noise
  img = cv2.resize(img, (200, 66)) #matches size used in NVIDIA model architecture
  img = img/255 #normalize image
  return img

try:
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    pwm_motor.ChangeDutyCycle(40)
    for frame in camera.capture_continuous(rawCapture,format="rgb",use_video_port=True):
        image=frame.array
        cv2.imshow("Frame",image)
        processed_img = img_preprocess(image)
        #cv2.imshow("Processed Frame",processed_img)
        processed_img = np.array([processed_img], dtype=np.float32)
        ####################3
        #global interpreter
        #global input_details
        #global output_details
        interpreter.set_tensor(input_details[0]['index'],processed_img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        #print(output_data)
        #global servos_pwm
        pwm_servo.ChangeDutyCycle(float(output_data))
        servos_pwm = output_data
        #steering_angle = float(interpreter.predict(image))
        #######
        key=cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        #print("Servo PWM:",str(servos_pwm),"and motor PWM:",str(motors_pwm))
        if key == ord("q"):
            break
except KeyboardInterrupt:
    pass

# Cleanup everything at end of file
cv2.destroyAllWindows()
camera.close()
pwm_servo.stop()
pwm_motor.stop()
GPIO.cleanup()