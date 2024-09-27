from gpiozero import DigitalInputDevice
from time import sleep


rain_sensor = DigitalInputDevice(17)
#chip = gpiod.Chip('gpiochip4')
#rain_sensor = 17
#rain_line = chip.get_line(rain_sensor)
#rain_line.request(consumer="relay1", type=gpiod.LINE_REQ_DIR_IN)

while True:
    if rain_sensor.is_active:
        print("no rain")
    else:
        print("rain")
    sleep(1)