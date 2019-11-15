from vad import VoiceActivityDetector
import sys

v = VoiceActivityDetector(sys.argv[1])
raw_detection = v.detect_speech()
speech_labels = v.convert_windows_to_readible_labels(raw_detection)
print(speech_labels)