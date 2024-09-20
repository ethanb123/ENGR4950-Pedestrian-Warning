from gpiozero import DistanceSensor
from time import sleep
ultrasonic = DistanceSensor(echo=17, trigger=4)
print("fdfhf")
while True:

    print(ultrasonic.distance)
    sleep(1)