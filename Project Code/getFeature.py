from os import path
from pprint import pprint
from pocketsphinx.pocketsphinx import *
from json import dump
from glob import glob

# from sphinxbase.sphinxbase import *

MODELDIR = "pocketsphinx/model"
DATADIR = "data/"
# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
config.set_string('-allphone', path.join(MODELDIR, 'en-us/en-us-phone.lm.dmp'))
config.set_float('-lw', 2.0)
config.set_float('-beam', 1e-10)
config.set_float('-pbeam', 1e-10)

name_list = ["hello", "on", "off", "go"]
group_len = .25
step_len = .15
all_phones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH',
              'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W',
              'Y', 'Z', 'ZH']  # Decode streaming data.
X = []
Y = []

for name in name_list:
	directory = DATADIR + "out_" + name
	for i in range(len(glob('./' + directory + '/*'))):
		decoder = Decoder(config)
		decoder.start_utt()
		stream = open(path.join(directory, name + str(i) + '.wav'), 'rb')
		while True:
			buf = stream.read(1024)
			if buf:
				decoder.process_raw(buf, False, False)
			else:
				break
		decoder.end_utt()

		final_list = []

		hyp = decoder.hyp()
		list_of_hyp = hyp.hypstr.split(" ")
		while "SIL" in list_of_hyp:
			list_of_hyp.remove("SIL")

		hyp_len = len(list_of_hyp)
		for i in range(int((1 - group_len) / step_len) + 1):
			sub_list = list_of_hyp[int(i * step_len * hyp_len): int((i * step_len + group_len) * hyp_len)]
			final_list.extend(
				# [format(round(sub_list.count(key) / float(len(sub_list)), 5), '.5f') for key in all_phones]
				[sub_list.count(key) / float(len(sub_list)) for key in all_phones]
			)
		X.append(final_list)
		Y.append(name)

with open("./output/out.txt", "w") as out:
	dump({"X": X, "Y": Y}, out)
# pprint({"X": X, "Y": Y}, out, width=5000, depth=3)
