import adafruit_ssd1306
import board
import busio
import digitalio
import speech_recognition as sr
import soundfile as sf
import librosa
from keras.models import load_model

# Input: librosa audio file
# Output: Formatted MFCC of the audio file
def librosa2mfcc(wave, max_len=11):
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc

# Returns prediction of CNN
def predict(file, model):
    sample = librosa2mfcc(file)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    predictions = model.predict(sample_reshaped)
    return LABELS[np.argmax(predictions)] if np.argmax(predictions) > MIN_CONFIDENCE else ''

# Runs multiple functions in parallel
def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

# To recognize speech
def speech():
	try:
		result = r.recognize_google(audio)
		if result == 'stop':
			oled.fill(0)
			oled.show()
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

# To recognize other sounds
def other_sounds():
	with open('tmp.wav','wb') as f:
        f.write(audio.get_wav_data())
	l_audio, _ = librosa.load('tmp.wav', mono=True, sr=None)
	oled.text(predict(l_audio, model), 1, 55, 1)

LENGTH = 6
HEIGHT = 9
MAX_CHARS = 128 // LENGTH
PATH_TO_MODEL = 'model.h5'
LABELS = ['Car Horn', 'Dog Bark', 'Gun Shot', 'Siren']

spi = busio.SPI(board.SCLK, MOSI=board.MOSI)
dc_pin = digitalio.DigitalInOut(board.D24)    # any pin!
reset_pin = digitalio.DigitalInOut(board.D23) # any pin!
cs_pin = digitalio.DigitalInOut(board.D22)    # any pin!
 
oled = adafruit_ssd1306.SSD1306_SPI(128, 64, spi, dc_pin, reset_pin, cs_pin)

oled.fill(0)
oled.text('Initializing...', 0, 0, 1)
oled.show()
print('Initializing...')

r = sr.Recognizer()
r.energy_threshold = 4000
r.dynamic_energy_threshold = True
r.dynamic_energy_adjustment_ratio = 1.3 # default 1.5; lower values lead to
                                        # more false positives and less false negatives
r.pause_threshold = 0.4 # default 0.8; how long to wait before recognizing;
                        # lower values lead to quicker recognition

print('Loading model...')
model = load_model(PATH_TO_MODEL)

oled.fill(0)
oled.text('Ready!', 0, 0, 1)
oled.show()
print('Ready!')

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
	print('Processing...')

	runInParallel(speech, other_sounds)
	
	oled.show()