
import gpiod
import time
import atexit

relay1 = 14
relay2 = 15
chip = gpiod.Chip('gpiochip4')
chip2 = gpiod.Chip('gpiochip4')
led_line = chip.get_line(relay1)
led_line2 = chip2.get_line(relay2)
led_line.request(consumer="relay1", type=gpiod.LINE_REQ_DIR_OUT)
led_line2.request(consumer="relay2", type=gpiod.LINE_REQ_DIR_OUT)

try:
    while True:
        led_line.set_value(1)
        time.sleep(0.5)
        led_line.set_value(0)
        led_line2.set_value(1)
        time.sleep(0.5)
        led_line2.set_value(0)
finally:
	led_line.set_value(0)
	led_line2.set_value(0)
	led_line.release()
	led_line2.release()

def exit_handler():
    led_line.set_value(0)
    led_line2.set_value(0)
    led_line.release()
    led_line2.release()

atexit.register(exit_handler)
