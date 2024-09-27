from gpiozero import MotionSensor
import time

pir = MotionSensor(14)
print("You moved111")
i=0

while pir.motion_detected:
		print("You moved: "+str(i))
		i+=1


while True:
	if pir.motion_detected:
		print("You moved: "+str(i))
		i+=1
		#pir.wait_for_no_motion()
		time.sleep(1)
	else:
		print("no motion")
		time.sleep(1)
#	pir.wait_for_motion()
#	print("You moved: "+str(i))
##	pir.wait_for_no_motion()
#	print("No Motion")
	

