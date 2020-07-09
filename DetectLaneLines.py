###############################
#### Stop Car at Stop Sign ####
#### --------------------- ####
#### Objective:		   ####
#### > Drive car forward   ####
####   until a sotpsign is ####
####   spotted on the cam  ####
###############################
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO
from time import sleep
import numpy as np
import cv2

# Inializations: #
#Camera fps/size
camera=PiCamera()
camera.resolution=(640,480)
camera.framerate=20
rawCapture=PiRGBArray(camera,size=(640,480))
sleep(0.1)
#Car PIN setups
enA=14
in1=15
in2=18
temp1=1
servo=22
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
pwm_servo.start(12.5)
pwm_motor.start(25)

def make_coordinate(img,line_parameters):
    slope,intercept=line_parameters
    y1=img.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(img,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        # returns slope 1st and y-intercept 2nd
        parameters = np.polyfit((x1,x2),(y1,y2),1)#degree 1
        slope=parameters[0]
        intercept=parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_coordinate(img, left_fit_average)
    right_line=make_coordinate(img, right_fit_average)
    return np.array([left_line, right_line])

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 #only one color because it is a gray image
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img,lines):
    img=np.copy(img)
    blank_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            #line(image,pt1,pt2,color,thickness=None,lineType=None,shift=None)
            cv2.line(blank_img,(x1,y1),(x2,y2),(0,255,0),thickness=4)
    img=cv2.addWeighted(img,0.8,blank_img,1,0.0)
    return img

def process(img):
    #image dimensions
	height = img.shape[0]
	width = img.shape[1]
	#Bottom triangle region
	#region_of_interest_vertices = [
	#    (0, height),
	#    (width/2, height/2),
	#    (width, height)
	#]
	#Bottom half of image
	region_of_interest_vertices = [
		(0, height),
		(0, height/2),
		(width, height/2),
		(width, height)
	]
	gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #Gaussian Blur will remove noise in image
    blur = cv2.GaussianBlur(gray_img, (5,5),0)
	#Canny(img,minThreshold,maxThreshold) 1:2 or 1:3
	canny_image = cv2.Canny(gray_img,50,150)
	cropped_image = region_of_interest(canny_image,
	                np.array([region_of_interest_vertices], np.int32),)
	lines = cv2.HoughLinesP(cropped_image,
                    #smaller rho/theta=more accurate longer processing time
                            rho=6, #number of pixels
                            theta=np.pi/60,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=25)
    #image_w_lines=draw_the_lines(img,lines)
    averaged_lines=average_slope_intercept(img,lines)
    image_w_lines=draw_the_lines(img,averaged_lines)

    return image_w_lines

# Begin Camera video and driving forward #
camera.start_preview()
for frame in camera.capture_continuous(rawCapture,format="rgb",use_video_port=True):
	# Begin driving at medium speed
	#pwm_motor.ChangeDutyCycle(50) #medium speed
    #pwm_motor.ChangeDutyCycle(80) #high speed
	pwm_motor.ChangeDutyCycle(70) #med-hi speed
	GPIO.output(in1,GPIO.HIGH)
	GPIO.output(in2,GPIO.LOW)
	pwm_servo.ChangeDutyCycle(12.5)
	# grab raw NumPy array representing image - 3D array
	image=frame.array
    #Find lanes in image region of interest and draw lines on them
    image = process(image)
	break
	# Wait 1 ms and read key input
	if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	#clear the stream in preparation for the next frame
	rawCapture.truncate(0)
# End camera functions
camera.stop_preview()
cv2.destroyAllWindows()
camera.close()
# End car functions
pwm_servo.stop()
pwm_motor.stop()
GPIO.cleanup()
