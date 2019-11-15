import adafruit_ssd1306
import board
import busio
import digitalio
import speech_recognition as sr
import os

LENGTH = 6
HEIGHT = 9
MAX_CHARS = 128 // LENGTH

spi = busio.SPI(board.SCLK, MOSI=board.MOSI)
dc_pin = digitalio.DigitalInOut(board.D24)    # any pin!
reset_pin = digitalio.DigitalInOut(board.D23) # any pin!
cs_pin = digitalio.DigitalInOut(board.D22)    # any pin!
 
oled = adafruit_ssd1306.SSD1306_SPI(128, 64, spi, dc_pin, reset_pin, cs_pin)

oled.fill(0)
oled.text('Initializing...', 0, 0, 1)
oled.show()

r = sr.Recognizer()
r.energy_threshold = 4000
r.dynamic_energy_threshold = True
r.dynamic_energy_adjustment_ratio = 1.3 # default 1.5; lower values lead to
                                        # more false positives and less false negatives
r.pause_threshold = 0.4 # default 0.8; how long to wait before recognizing;
                        # lower values lead to quicker recognition

oled.fill(0)
oled.text('Ready!', 0, 0, 1)
oled.show()

while True:
	with sr.Microphone() as source:
		oled.text('Calibrating...', 0, 56, 1)
		oled.show()
		r.adjust_for_ambient_noise(source, duration=0.5)
		oled.text('Calibrating...', 0, 56, 0)
		oled.show()
		audio = r.listen(source)
	oled.fill(0)
	oled.text('Processing...', 0, 0, 1)
	oled.show()
	oled.fill(0)
	try:
		result = r.recognize_google(audio)
		if result == 'stop':
			oled.fill(0)
			oled.show()
			quit()
		elif result == 'turn off':
			oled.fill(0)
			oled.show()
			os.system("sudo shutdown now -h")
			quit()
		print(result)
		words = result.split(' ')
		chars = 0
		row = 0
		for word in words:
			word += ' '
			if len(word) + 1 + chars <= MAX_CHARS:
				if chars == 0:
					oled.text(word, 0, row * HEIGHT, 1)
				else:
					oled.text(word, chars * LENGTH, row * HEIGHT, 1)
				chars += len(word)
			else:
				row += 1
				chars = len(word)
				oled.text(word, 0, row * HEIGHT, 1)
	except sr.UnknownValueError:
		oled.text('Couldn\'t recognize', 0, 0, 1)
		oled.text('speech', 0, 9, 1)
		print('Couldn\'t recognize speech')

	oled.show()