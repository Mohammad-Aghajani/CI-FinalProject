#! /usr/bin/python
# coding=utf-8
from os import path
from numpy import array, matmul, append, array_equal
from pocketsphinx.pocketsphinx import *
from json import load
from math import tanh
import pyaudio
import wave
from PyQt4.QtGui import QMainWindow, QApplication, QWidget, QFileDialog
from UIMain import Ui_MainWindow

try:
	_encoding = QApplication.UnicodeUTF8


	def _translate(context, text, disambig):
		return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
	def _translate(context, text, disambig):
		return QApplication.translate(context, text, disambig)

MODELDIR = "pocketsphinx/model"
DATADIR = "data"
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2.5
audio = pyaudio.PyAudio()

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
	# if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
	print "Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name')

sigma = lambda x: tanh(x)
with open('output/weight.txt') as file:
	data = load(file)
	w = array(data["W"])
	v = array(data["V"])

name_list = {"hello": [1, -1, -1, -1], "on": [-1, 1, -1, -1], "off": [-1, -1, 1, -1], "go": [-1, -1, -1, 1]}

group_len = .25
step_len = .15

all_phones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH',
              'IY',
              'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W',
              'Y',
              'Z', 'ZH']  # Decode streaming data.


class MainWindow(QMainWindow):
	def __init__(self, parent=None):
		QWidget.__init__(self, parent)
		# Set up the user interface from Designer.
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.speak.clicked.connect(self.start_cb)
		self.ui.file.clicked.connect(self.openWaveFile)

	def start_cb(self):
		self.record()

	def openWaveFile(self):
		dlg = QFileDialog()
		dlg.setFileMode(QFileDialog.AnyFile)
		dlg.setFilter("Wave File(*.wav)")

		if dlg.exec_():
			filenames = dlg.selectedFiles()
			with open(filenames[0], 'rb') as stream:
				config = Decoder.default_config()
				config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
				config.set_string('-allphone', path.join(MODELDIR, 'en-us/en-us-phone.lm.dmp'))
				config.set_float('-lw', 2.0)
				config.set_float('-beam', 1e-10)
				config.set_float('-pbeam', 1e-10)
				decoder = Decoder(config)
				decoder.start_utt()

				while True:
					buf = stream.read(1024)
					if buf:
						decoder.process_raw(buf, False, False)
					else:
						break
				decoder.end_utt()

				test = []

				hyp = decoder.hyp()
				list_of_hyp = hyp.hypstr.split(" ")
				while "SIL" in list_of_hyp:
					list_of_hyp.remove("SIL")

				hyp_len = len(list_of_hyp)
				for i in range(int((1 - group_len) / step_len) + 1):
					sub_list = list_of_hyp[int(i * step_len * hyp_len): int((i * step_len + group_len) * hyp_len)]
					test.extend(
						# [format(round(sub_list.count(key) / float(len(sub_list)), 5), '.5f') for key in all_phones]
						[sub_list.count(key) / float(len(sub_list)) for key in all_phones]
					)

				y = list(map(sigma, matmul(append(test, array([-1])), v.T)))
				y.append(-1)
				z = list(map(sigma, matmul(y, w.T)))
				max_z = max(z)
				out = [1 if d is max_z else -1 for d in z]
				print("result is = ", out)
				if array_equal(out, name_list["hello"]):
					self.ui.textEdit.append(_translate("MainWindow", "سلام", None))
				elif array_equal(out, name_list["off"]):
					self.ui.textEdit.append(_translate("MainWindow", "خاموش", None))
				elif array_equal(out, name_list["on"]):
					self.ui.textEdit.append(_translate("MainWindow", "روشن", None))
				elif array_equal(out, name_list["go"]):
					self.ui.textEdit.append(_translate("MainWindow", "برو", None))

	def record(self):
		# Create a decoder with certain model
		config = Decoder.default_config()
		config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
		config.set_string('-allphone', path.join(MODELDIR, 'en-us/en-us-phone.lm.dmp'))
		config.set_float('-lw', 2.0)
		config.set_float('-beam', 1e-10)
		config.set_float('-pbeam', 1e-10)

		# start Recording
		stream = audio.open(format=FORMAT, channels=CHANNELS,
		                    rate=RATE, input=True,
		                    input_device_index=8,
		                    frames_per_buffer=CHUNK)

		print ("recording...")

		decoder = Decoder(config)
		decoder.start_utt()

		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			buf = stream.read(1024)
			if buf:
				decoder.process_raw(buf, False, False)

		decoder.end_utt()
		print ("finished recording")
		# stop Recording
		stream.stop_stream()
		stream.close()

		test = []

		hyp = decoder.hyp()
		list_of_hyp = hyp.hypstr.split(" ")
		while "SIL" in list_of_hyp:
			list_of_hyp.remove("SIL")

		hyp_len = len(list_of_hyp)
		for i in range(int((1 - group_len) / step_len) + 1):
			sub_list = list_of_hyp[int(i * step_len * hyp_len): int((i * step_len + group_len) * hyp_len)]
			test.extend(
				# [format(round(sub_list.count(key) / float(len(sub_list)), 5), '.5f') for key in all_phones]
				[sub_list.count(key) / float(len(sub_list)) for key in all_phones]
			)

		y = list(map(sigma, matmul(append(test, array([-1])), v.T)))
		y.append(-1)
		z = list(map(sigma, matmul(y, w.T)))
		max_z = max(z)
		out = [1 if d is max_z else -1 for d in z]
		print("result is = ", out)
		if array_equal(out, name_list["hello"]):
			self.ui.textEdit.append(_translate("MainWindow", "سلام", None))
		elif array_equal(out, name_list["off"]):
			self.ui.textEdit.append(_translate("MainWindow", "خاموش", None))
		elif array_equal(out, name_list["on"]):
			self.ui.textEdit.append(_translate("MainWindow", "روشن", None))
		elif array_equal(out, name_list["go"]):
			self.ui.textEdit.append(_translate("MainWindow", "برو", None))


if __name__ == '__main__':
	app = QApplication([])
	winndow = MainWindow()
	winndow.show()
	app.exec_()
