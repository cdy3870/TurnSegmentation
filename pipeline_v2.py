# from parse_transcript import *
from transcriptValidator import *
import stanfordnlp
import nltk
import pickle 
import numpy
from nltk.parse import CoreNLPParser
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import os
import json
import itertools
import csv
import re
from sklearn.metrics import confusion_matrix
from unidecode import unidecode
import difflib as d1
import collections
import copy
import wordninja

# To be used globally
curly_list = []
brack_list = []
paren_list = []
file_dict = {}

def get_gold_stats(file_dir):
	"""
	Prints the gold segmentation and token statistics given the dataset

	file_dir: the data file path

	returns: None
	"""

	encodings_y_list = []
	full_arg_list = []
	full_seg_list = []
	num_segs_list = []
	full_combined_tokens = []

	files = [os.path.join(file_dir, filename) for filename in os.listdir(file_dir) if filename.endswith(".xlsx")]	
	for file in files:
		out = convert_transcript(file)
		encodings_y, remove_list, encoding_lens, segments_list = encode_y(to_TurnSegmentation(convert_transcript(file)))
		arg_list, tokenized_arg_list, arg_list_lens = make_arg_list(to_TurnSegmentation(convert_transcript(file)), file_name=file[16:])
		full_arg_list += tokenized_arg_list
		for i in range(len(tokenized_arg_list)):
			combined_s = []
			list(map(combined_s.extend, segments_list[i]))
			full_seg_list.append(combined_s)
		insert_Os(tokenized_arg_list, segments_list, encodings_y, file, None, out, write_to_file=False)
		encodings_y_list += encodings_y
		combined_tokens = []
		list(map(combined_tokens.extend, encodings_y))
		full_combined_tokens += combined_tokens
		num_segs_list += gold_counter_y(to_TurnSegmentation(convert_transcript(file)), file)

	print(str(file_dir) + " Gold Segmentation Statistics (With O-insertions)")
	print("Gold Segment Counts: ", end='')
	print(collections.Counter(num_segs_list))
	print("Counter Segment Counts: ", end='')
	print(collections.Counter(segment_counter_2(encodings_y_list)))
	print("Counter Token Counts: ", end='')
	print(collections.Counter(full_combined_tokens))
	print("Bigrams: ", end='')
	print(generate_bigrams(encodings_y_list))


def generate_trigrams(encodings):
	"""
	Prints the trigrams given a list of encodings

	encodings: the encodings of the turns

	returns: a dict containing the trigrams
	"""
	
	trigram_dict = {}
	for encoding in encodings:
		encoding = num_to_tag(encoding)
		for x in range(len(encoding) - 2):
			str_concatenated = encoding[x] + encoding[x+1] + encoding[x+2]
			# if str_concatenated == "BB":
			# 	print(encoding)
			if str_concatenated in trigram_dict.keys():
				trigram_dict[str_concatenated] += 1
			else:
				trigram_dict[str_concatenated] = 1

	return trigram_dict

def generate_bigrams(encodings):
	"""
	Prints the bigrams given a list of encodings

	encodings: the encodings of the turns

	returns: a dict containing the bigrams
	"""

	bigram_dict = {}
	for encoding in encodings:
		encoding = num_to_tag(encoding)
		for x in range(len(encoding) - 1):
			str_concatenated = encoding[x] + encoding[x+1]

			# Listing encoding for BB edge case
			# if str_concatenated == "BB":
				# print(encoding)

			if str_concatenated in bigram_dict.keys():
				bigram_dict[str_concatenated] += 1
			else:
				bigram_dict[str_concatenated] = 1

	return bigram_dict

def check_for_counts_thresholding(encodings, counts_x, arg_list, file_name):
	"""
	Prints the encoding, turn, and raw text where the segment count is bounded,
	used when iterating through file directory (to print which file)

	encodings: the encodings of the turns
	counts_x: the counts of segments that have been bounded
	arg_list: the raw turns
	file_name: the file in which occurs

	returns: None
	"""
	for x in range(len(encodings)):
		if collections.Counter(encodings[x])[0] != counts_x[x]:
			print("File: " + str(file_name))
			print("Turn Num: " + str(x + 1))
			print("Raw Text: ", end="")
			print(arg_list[x])
			print("Encoding: ", end="")
			print(num_to_tag(encodings[x]))
			print("\n")		

def replace_lower_postprocess(encoding, counts_x, index, lower_bound):
	"""
	Postprocess the predicted encoding to meet the lower boundard requirement

	encoding: the encoding of the turn
	counts_x: the counts of segments that have been unbounded
	index: the index of the particular turn
	lower_bound: the bound to meet

	returns: None
	"""

	if counts_x[index] < lower_bound:
		# print("Before")
		# print(num_to_tag(encoding))
		encoding[0] = 0
		# print("After")
		# print(num_to_tag(encoding))
		# print("\n")

def replace_upper_postprocess(encoding, counts_x, index, upper_bound):	
	"""
	Postprocess the predicted encoding to meet the upper boundard requirement

	encoding: the encoding of the turn
	counts_x: the counts of segments that have been unbounded
	index: the index of the particular turn
	upper_bound: the bound to meet

	returns: None
	"""

	if counts_x[index] > upper_bound:
		num_removals = counts_x[index] - upper_bound
		counts_x[index] = upper_bound
		# print("Before")
		# print(num_to_tag(encoding))
		for i in range(len(encoding) - 1, -1, -1):
			if encoding[i] == 0:
				encoding[i] = 2
				num_removals -= 1
			if num_removals == 0:
				# print("After")
				# print(num_to_tag(encoding))
				# print("\n")
				break

def bigram_postprocess(encoding):
	"""
	Postprocess the predicted encoding to remove "BB" instandces

	encoding: the encoding of the turn

	returns: None
	"""

	for i in range(len(encoding) - 1):
		if num_to_tag(encoding)[i] + num_to_tag(encoding)[i + 1] ==  "BB":
			# print("Before")
			# print(num_to_tag(encoding))
			encoding[i + 1] = 2
			# print("After")
			# print(num_to_tag(encoding))
			# print("\n")

def num_to_tag(encoding):
	"""
	Convert from numerical (210) labeling to IOB labeling

	encoding: the encoding of the turn

	returns: a deepcopy of the encoding
	"""

	return_encoding = copy.deepcopy(encoding)
	encoding_map = {0:"B", 2:"I", 1:"O"}
	for i in range(len(encoding)): 
		return_encoding[i] = encoding_map[float(encoding[i])]

	return return_encoding

def tag_to_num(encoding):
	"""
	Convert from IOB labeling to numerical (210) labeling

	encoding: the encoding of the turn

	returns: a deepcopy of the encoding
	"""

	return_encoding = copy.deepcopy(encoding)
	encoding_map = {"B":0, "I":2, "O":1}
	for i in range(len(encoding)): 
		return_encoding[i] = encoding_map[encoding[i]]
	
	return return_encoding

def verify(segment_mat, token_mat, include_zero=False):
	"""
	Verifies that the segment-level and token-level confusion matrices line up,
	won't line up if there is bounding, predictions would need postprocessing

	segment_mat: the output of sklearn's confusion matrix using segment counts
	token_mat: the output of sklearn's confusion matrix using token counts
	include_zero: true if counts start at zero, false otherwise

	returns: None
	"""

	segment_pred_sum = 0
	token_pred_sum = 0
	segment_true_sum = 0
	token_true_sum = 0

	if include_zero:
		increment = 0
	else:
		increment = 1


	for i in range(len(segment_mat)): segment_pred_sum += ((i + increment) * np.sum(segment_mat[i]))
	# segment_pred_sum = np.sum(segment_mat[0]) + 2*np.sum(segment_mat[1]) + 3*np.sum(segment_mat[2]) + 4*np.sum(segment_mat[3])
	token_pred_sum = np.sum(token_mat[0])
	print("Segment Pred Sum: " + str(segment_pred_sum))
	print("Token Pred Sum: " + str(token_pred_sum))

	for i in range(len(segment_mat)): segment_true_sum += ((i + increment) * segment_mat[:, i].sum())
	# segment_true_sum = segment_mat[:, 0].sum() + 2*segment_mat[:, 1].sum() + 3*segment_mat[:, 2].sum() + 4*segment_mat[:, 3].sum()
	token_true_sum = token_mat[:, 0].sum()
	print("Segment True Sum: " + str(segment_true_sum))
	print("Token True Sum: " + str(token_true_sum))



def segment_counter_4(encodings):
	"""
	Segment counter method that has an upper bound for predictions

	encodings: the encoded turns

	return: the segment counts
	"""

	num_segments_list = [collections.Counter(x)[0] if collections.Counter(x)[0] < 4 else 4 for x in encodings]

	return num_segments_list

def segment_counter_3(encodings):
	"""
	Segment counter method that has a lower bound for predictions

	encodings: the encoded turns

	return: the segment counts
	"""

	num_segments_list = [collections.Counter(x)[0] if collections.Counter(x)[0] != 0 else 1 for x in encodings]

	return num_segments_list

def segment_counter_2(encodings):
	"""
	Segment counter method with no bounding for predictions
	
	encodings: the encoded turns

	return: the segment counts
	"""

	num_segments_list = [collections.Counter(x)[0] for x in encodings]

	return num_segments_list

def segment_counter_1(encodings):
	"""
	Segment counter method with upper and lower bounds for predictions
	
	encodings: the encoded turns

	return: the segment counts
	"""

	num_segments_list = []
	count = 0
	for x in encodings:
		count = collections.Counter(x)[0]


		if count == 0:
			count = 1

		if count > 4:
			count = 4

		num_segments_list.append(count)

	return num_segments_list


def gold_counter_x(encodings):
	"""
	Segment counter method for predictions that returns counts 
	with and without bounding, actually used in the pipeline
	
	encodings: the encoded turns

	return: bounded_num_segments_list- segment counts with bounds
			unbounded_num_segments_list- segment counts without bounds
	"""

	bounded_num_segments_list = []
	unbounded_num_segments_list = []
	# enc_list = []
	count = 0
	for x in encodings:
		count = int(collections.Counter(x)[0])

		unbounded_num_segments_list.append(count)

		if count == 0:
			count = 1

		if count > 3:
			count = 3

		bounded_num_segments_list.append(count)

	return bounded_num_segments_list, unbounded_num_segments_list

def gold_counter_y(collab, file):
	"""
	Segment counter for gold segmentation, used on raw segmentation

	collab: parse_transcript formatted output
	file: file name of transcript

	return: the segment counts
	"""

	num_segments_list = []
	count = 0
	for x in collab:
		for y in x["Argumentation"]:
			if y != '' and y != "None":
				count += 1
		num_segments_list.append(count)
		count = 0

	return num_segments_list

def print_segments(encoding_2, encoding_1, tokenized_sentence, opened_file, write_to_file=False):
	"""
	Prints segments during decoding for both gold segmentation and predictions

	encoding_2: the encoded turn
	encoding_1: the predicted encoding
	tokenized_sentence: the tokenized and preprocessed turn
	opened_file: the file where decodings are written to
	write_to_file: a flag to determine if decodings are written to

	return: None
	"""


	# print("\n")
	segment = []
	stack = []
	# print("Gold Standard")
	# print(num_to_tag(encoding_1))

	if write_to_file:
		opened_file.writelines(["\nGold Standard: "])
		write_list(num_to_tag(encoding_1), opened_file, ignore_t=True)

	count = 0

	for i in range(len(encoding_1)):	
		if i == 0:
			segment = []
			segment.append(tokenized_sentence[i])
			count += 1
		elif encoding_1[i] == 2:
			segment.append(tokenized_sentence[i])
		elif encoding_1[i] == 0:
			# print("Segment " + str(count) + ": ", end = '')
			# print(segment)

			if write_to_file:
				opened_file.writelines(["Segment " + str(count) + ": "])
				write_list(segment, opened_file, ignore_t=True)

			segment = []
			segment.append(tokenized_sentence[i])
			count += 1
			# if count > 1:
			# 	key_words_y.append(tokenized_sentence.lower())

		if i == len(encoding_1) - 1:
			# print("Segment " + str(count) + ": ", end = '')
			# print(segment)

			if write_to_file:
				opened_file.writelines(["Segment " + str(count) + ": "])
				write_list(segment, opened_file, ignore_t=True)

	# print("\n")

	# Only create segment if 0 and 2 exists in stack
	# print("Predicted Segments")
	# print(num_to_tag(encoding_2))

	if write_to_file:
		opened_file.writelines(["\nPredicted Segments: "])
		write_list(num_to_tag(encoding_2), opened_file, ignore_t=True)

	segment = []
	stack = []

	count = 0

	for i in range(len(encoding_2)):	
		if i == 0:
			segment = []
			segment.append(tokenized_sentence[i])
			count += 1
			# stack = []
			# stack.append(test_x_encodings[j][i])
			# check_append(segment, tokenized_sentence[i])
			# print("1")
		elif encoding_2[i] == 2:
			segment.append(tokenized_sentence[i])
			# check_append(segment, tokenized_sentence[i])
		elif encoding_2[i] == 0:
			# print("Segment " + str(count) + ": ", end = '')
			# print(segment)

			if write_to_file:
				opened_file.writelines(["Segment " + str(count) + ": "])
				write_list(segment, opened_file, ignore_t=True)

			segment = []
			segment.append(tokenized_sentence[i])
			count += 1

			# if count_x > 1:
			# 	key_words_x.append(tokenized_sentence.lower())

			# stack = []
			# stack.append(test_x_encodings[j][i])
			# check_append(segment, tokenized_sentence[i])
			# pred_segment_count += 1

		if i == len(encoding_2) - 1:
			# print("Segment " + str(count) + ": ", end = '')
			# print(segment)

			if write_to_file:
				opened_file.writelines(["Segment " + str(count) + ": "])
				write_list(segment, opened_file, ignore_t=True)

			# sub_segment_list.append(segment)
			# pred_segment_count += 1

def decode_x(test_x_encodings, test_y_encodings, test_arg_list, segment_list, opened_file=None, write_to_file=False, file_name=None):
	"""
	Decode predictions and print or save the decodings

	test_x_encodings: the predicted encodings
	test_y_encodings: the encoded turns
	test_arg_list: the raw turns at talk
	segment_list: 
	opened_file: the file where decodings are written to
	write_to_file: a flag to determine if decodings are written to
	file_name: the file name of the transcript

	return: None
	"""

	# print(test_x_encodings[1])
	test = []
	segment = []
	true_segment_count = 0
	pred_segment_count = 0
	desired_correct_segments = {1:0, 2:0, 3:0, 4:0}
	num_correct_segments = {1:0, 2:0, 3:0, 4:0}
	temp_dict = {"11":0, "12": 0, "13": 0, "14": 0, "21":0, "22": 0, "23": 0, "24": 0, "31":0, "32": 0, "33": 0, "34": 0, "41":0, "42": 0, "43": 0, "44": 0}
	sub_segment_list = []

	if write_to_file:
		opened_file.writelines(["-------------------------------------------- " + str(file_name) + " ----------------------------------------------------"])

	# print("-------------------------------------------- " + str(file_name) + " ----------------------------------------------------")
	for j in range(len(test_x_encodings)):
		if write_to_file:
			opened_file.writelines(["-------------------------------------------- " + str(j) + " ----------------------------------------------------\n"])
			opened_file.writelines("Raw Text: " + str(test_arg_list[j]))

		# print("-------------------------------------------- " + str(j) + " ----------------------------------------------------")
		# print("Raw Text: ", end = '')
		# print(test_arg_list[j])

		tokenized_sentence = preprocess_chunk(test_arg_list[j])

		print_segments(test_x_encodings[j], test_y_encodings[j], tokenized_sentence, opened_file, write_to_file)


def class_report_segment_level(counts_x, counts_y, path, write_to_csv=True):
	"""
	Prints the classification report on the segment level

	counts_x: the predicted number of segments derived from a segment counting method
	counts_y: the gold segmentation counts
	path: the csv file where the report is written to
	write_to_csv: a flag that determines if the report is written to a csv file

	return: None
	"""

	target_names = ["1", "2", "3"]
	target_names_2 = ["1", "2", "3", "4"]
	target_names_3 = ["1", "2", "3", "4", "5"]
	# target_names = ["0", "1", "2", "3", "4", "5",]


	try:
		report = classification_report(counts_y, counts_x, target_names=target_names, output_dict=True)
	except:
		try:
			report = classification_report(counts_y, counts_x, target_names=target_names_2, output_dict=True)
		except:
			report = classification_report(counts_y, counts_x, target_names=target_names_3, output_dict=True)

	# report = classification_report(counts_y, counts_x, target_names=target_names_3, output_dict=True)
	print(report)

	if write_to_csv:
		with open(path, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"])            
			csv_writer.writerow(["Class Report Segment Level"])
			csv_writer.writerow(["pred-true", "1", "2", "3 or more"])
			csv_writer.writerow(["Precision", report["1"]["precision"], report["2"]["precision"], report["3"]["precision"]])
			csv_writer.writerow(["Recall", report["1"]["recall"], report["2"]["recall"], report["3"]["recall"]])
			csv_writer.writerow(["F-score", report["1"]["f1-score"], report["2"]["f1-score"], report["3"]["f1-score"]])


def conf_mat_segment_level(counts_x, counts_y, path, write_to_csv=True):
	"""
	Returns a confusion matrix of the segment level

	counts_x: the predicted number of segments derived from a segment counting method
	counts_y: the gold segmentation counts
	path: the csv file where the report is written to
	write_to_csv: a flag that determines if the report is written to a csv file

	return: the confusion matrix
	"""

	matrix_sum = confusion_matrix(counts_x, counts_y)	
	print(matrix_sum)

	if write_to_csv:
		with open(path, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"])            
			csv_writer.writerow(["Confusion Matrix Segment Level"])
			csv_writer.writerow(["pred-true", "1", "2", "3 or more"])
			csv_writer.writerow(["1", matrix_sum[0][0], matrix_sum[0][1], matrix_sum[0][2]])
			csv_writer.writerow(["2", matrix_sum[1][0], matrix_sum[1][1], matrix_sum[1][2]])
			csv_writer.writerow(["3 or more", matrix_sum[2][0], matrix_sum[2][1], matrix_sum[2][2]])
			csv_writer.writerow(["Sums", matrix_sum[:, 0].sum(), matrix_sum[:, 1].sum(), matrix_sum[:, 2].sum()])

	return np.array(matrix_sum)

def conf_mat_token_level(encodings_x, encodings_y, path, write_to_csv=True):
	"""
	Returns a confusion matrix of the token level

	encodings_x: the predicted encodings
	encodings_y: the encoded turns
	path: the csv file where the matrix is written to
	write_to_csv: a flag that determines if the matrix is written to a csv file

	return: the confusion matrix
	"""

	combined_y = []
	combined_x = []
	list(map(combined_y.extend, encodings_y))
	list(map(combined_x.extend, encodings_x))

	matrix_sum = confusion_matrix(combined_x, combined_y)
	print(matrix_sum)

	if write_to_csv:
		with open(path, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"])            
			csv_writer.writerow(["Confusion Matrix Token Level"])
			csv_writer.writerow(["pred-true", "B", "I", "O"])
			csv_writer.writerow(["B", matrix_sum[0][0], matrix_sum[0][2], matrix_sum[0][1]])
			csv_writer.writerow(["I", matrix_sum[2][0], matrix_sum[2][2], matrix_sum[2][1]])
			csv_writer.writerow(["O", matrix_sum[1][0], matrix_sum[1][2], matrix_sum[1][1]])
			csv_writer.writerow(["Sums", matrix_sum[:, 0].sum(), matrix_sum[:, 2].sum(), matrix_sum[:, 1].sum()])

	return np.array(matrix_sum)

def class_report_token_level(encodings_x, encodings_y, path, write_to_csv=True):
	"""
	Prints the classification report on the token level

	encodings_x: the predicted encodings
	encodings_y: the encoded turns
	path: the csv file where the matrix is written to
	write_to_csv: a flag that determines if the matrix is written to a csv file

	return: None
	"""

	total_samples = len(encodings_y)


	target_names = ["0-beginning", "1-padding", "2-inner"]
	# target_names_2 = ["2-inner", "1-padding"]
	# target_names_3 = ["2-inner"]

	# target_names = ["0-beginning", "2-inner"]


	combined_y = []
	combined_x = []
	list(map(combined_y.extend, encodings_y))
	list(map(combined_x.extend, encodings_x))

	# print(combined_x)
	# print(combined_y)
	try:
		report = classification_report(combined_y, combined_x, target_names=target_names, output_dict=True)
	except:
		try:
			report = classification_report(combined_y, combined_x, target_names=target_names_2, output_dict=True)
		except:
			report = classification_report(combined_y, combined_x, target_names=target_names_3, output_dict=True)

	# report = classification_report(combined_y, combined_x, target_names=target_names, output_dict=True)
	print(report)

	if write_to_csv:
		with open(path, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"])            
			csv_writer.writerow(["Class Report Token Level"])
			csv_writer.writerow(["pred-true", "B", "I", "O"])
			csv_writer.writerow(["Precision", report['0-beginning']["precision"], report['2-inner']["precision"], report['1-padding']["precision"]])
			csv_writer.writerow(["Recall", report['0-beginning']["recall"], report['2-inner']["recall"], report['1-padding']["recall"]])
			csv_writer.writerow(["F-score", report['0-beginning']["f1-score"], report['2-inner']["f1-score"], report['1-padding']["f1-score"]])


def check_multi(encodings_y):
	segment_count = 0
	for x in encodings_y:
		if x == 0:
			segment_count += 1
	# print(encodings_y)
	if segment_count > 1:
		# print("is multi")
		return True
	# print("is not multi")
	return False


def encode_y(collab, only_multi=False): # Commented out because lists already pickled
	#################### Create list of segments and encoded tokens #################################
	rows = 5 # Num rows in a turn
	segments = 3 # Num possible segments in a turn
	encodings_y = []
	token_list = []
	#################################### Format and encode y set ######################################
	sub_pos_list = []
	sub_sub_pos_list = []
	pos_list = []
	encoding_lens = []
	lens_sum = 0

	# stanfordnlp.download('en')
	# nlp = stanfordnlp.Pipeline()
	# empty_count = 0
	# empty_check = True
	# print(collab)
	# print(collab)
	for x in collab:
		for y in x["Argumentation"]:
			if y != '':
				doc = preprocess_chunk(y)
				for token in doc:
					sub_sub_pos_list.append(token)

				lens_sum += len(sub_sub_pos_list)
				sub_pos_list.append(sub_sub_pos_list)
				sub_sub_pos_list = []

				# When pos tags are needed
				# for i, sent in enumerate(doc.sentences):
				#     for word in sent.words:
				#         sub_sub_pos_list.append(word.pos)
				# sub_pos_list.append(sub_sub_pos_list)
				# sub_sub_pos_list = []

		encoding_lens.append(lens_sum)
		lens_sum = 0
		pos_list.append(sub_pos_list)
		sub_pos_list = []

	# print(pos_list)

	sub_encodings_y = []
	encodings_y = []
	remove_list = []

	for x in range(len(pos_list)):
		for y in range(len(pos_list[x])):
			for z in range(len(pos_list[x][y])):
				if z == 0:
					sub_encodings_y.append(0)
				else:
					sub_encodings_y.append(2)
		if only_multi:
			if check_multi(sub_encodings_y):
				encodings_y.append(sub_encodings_y)
			else:
				remove_list.append(x)		
		else:
			if collections.Counter(sub_encodings_y)[0] == 0:
				print(pos_list[x])
			encodings_y.append(sub_encodings_y)
		
		sub_encodings_y = []

	return encodings_y, remove_list, encoding_lens, pos_list


def make_arg_list(collab, remove_list=None, only_multi=False, file_name=""): # Commented out because lists already pickled
	###################### Create list of arguments and list of # of sentences ########################
	# print("collab: " + str(len(collab)))
	arg_list = []
	tokenized_arg_list = []
	# num_tokens_list = [0 for x in range(len(collab)) for y in range(len(nltk.word_tokenize(collab[x])))]
	num_sentences_list = []
	num_sentences = 0
	num_one_segment = 0
	count = 0
	arg_list_lens = []

	# print("collab: " + str(len(collab)))
	# print("remove list")
	# print(remove_list)
	for x in range(len(collab)):
		if only_multi:
			if count < len(remove_list):
				if x != remove_list[count]:			
					arg_list.append(unidecode(collab[x]["RawText"]))
					tokenized_arg_list.append(preprocess_chunk(collab[x]["RawText"]))
				else:
					count += 1
					pass
			else:
				arg_list.append(unidecode(collab[x]["RawText"]))
				tokenized_arg_list.append(preprocess_chunk(collab[x]["RawText"]))
		else:
			arg_list.append(unidecode(collab[x]["RawText"]))
			tokenized_arg_list.append(preprocess_chunk(collab[x]["RawText"], file_name))
			arg_list_lens.append(len(tokenized_arg_list[-1]))

	# print("arg: " + str(len(arg_list)))
	# print(arg_list)



		# Count the number of periods/question marks
		# num_sentences = len(arg_list[x]) - len(arg_list[x].replace(".", "")) \
		# 				+ len(arg_list[x]) - len(arg_list[x].replace("?", "")); 
		# # If no periods/question marks then consider as 1 sentence
		# if num_sentences == 0: 
		# 	num_sentences_list.append(1);
		# 	# num_segments_list.append(1);
		# 	num_one_segment += 1
		# else: 
		# 	num_sentences_list.append(num_sentences);
		# 	# num_segments_list.append(0);

	# print(arg_list)
	# Arg List
	# if write_enable == 1:
	# 	with open(os.path.join(cv_dir, arg_list_fn), 'wb') as filehandle:
	# 		pickle.dump(arg_list, filehandle)

	return arg_list, tokenized_arg_list, arg_list_lens


def write_list(list_name, opened_file, ignore_t=False):
	count = 0
	for item in list_name:
		if ignore_t:
			opened_file.write("%s " % item)
		else:
			opened_file.write("T" + str(count) + "- %s  " % item)
		count += 1
	opened_file.write("\n")

def check_space(combined_s, encoding, tokenized_arg, diff_list, file_name, opened_file, turn_num, write_to_file):
	"""
	Modifies the encoding accordingly if it's a space edge case (missing space)

	combined_s: the gold segmentation of the turns that have been unsegmented
	encoding: the encoded turn
	tokenized_arg: the preprocessed, tokenized turn
	diff_list: the python diff between the encoding and tokenized_arg
	file_name: the file name of the transcript
	opened_file: the edge case file handle written to
	turn_num: the turn number in the transcript
	write_to_file: a flag that determines if the edge case file is written to

	return: None
	"""

	if '?      ---\n' in diff_list:
		# Write to file
		if write_to_file:
			line = ["File: " + file_name + "\n", "Turn Num: " + turn_num + "\n", "Error: Missing Space in Segmented \n"]
			opened_file.writelines(line)
			opened_file.writelines(["Raw text: "])
			write_list(tokenized_arg, opened_file)
			opened_file.writelines(["Segmented text: "])
			write_list(combined_s, opened_file)

		split_list = []
		for k in range(len(combined_s)):
			temp_list = wordninja.split(combined_s[k])
			if len(temp_list) > 1:
				encoding.insert(k + 1, 2)
				split_list += temp_list
			elif len(temp_list) == 1:
				split_list += temp_list
			else:
				split_list.append(combined_s[k])

		split_list = [wordninja.split(token) if len(wordninja.split(token)) > 0 else token for token in combined_s]
		combined_s = list(itertools.chain.from_iterable(split_list))

		# print(encoding)

		diff_list = list(d1.Differ().compare(combined_s, tokenized_arg))
		# print(diff_list)

		if write_to_file:
			opened_file.writelines(["Resultant encoding: "])
			write_list(num_to_tag(encoding), opened_file, ignore_t=True)
			opened_file.write("\n")

		return True

	return False

	# line = [file_name]
	# opened_file.writelines(line)

def check_copy(combined_s, encoding, tokenized_arg, diff_list, file_name, opened_file, turn_num, write_to_file):
	"""
	Modifies the encoding accordingly if it's a copy edge case (copying error)

	combined_s: the gold segmentation of the turns that have been unsegmented
	encoding: the encoded turn
	tokenized_arg: the preprocessed, tokenized turn
	diff_list: the python diff between the encoding and tokenized_arg
	file_name: the file name of the transcript
	opened_file: the edge case file handle written to
	turn_num: the turn number in the transcript
	write_to_file: a flag that determines if the edge case file is written to

	return: None
	"""

	# Copy error edge case, save where this happens in an external file with transcript, turn number, raw text, segmented, resultant encoding, edge case type
	sub_count = 0
	threshold = 5
	# print(diff_list)
	for l in diff_list:
		if l[0] == "-": sub_count += 1
	if sub_count > threshold:
		# Write to file
		if write_to_file:
			line = ["File: " + file_name + "\n", "Turn Num: " + turn_num + "\n", "Error: Copy Error \n"]
			opened_file.writelines(line)
			opened_file.writelines(["Raw text: "])
			write_list(tokenized_arg, opened_file)
			opened_file.writelines(["Segmented text: "])
			write_list(combined_s, opened_file)

		combined_s = copy.deepcopy(tokenized_arg)

		# Re-encode, original segmentation is now missing
		encoding[:] = [2 for token in combined_s]
		encoding[0] = 0

		if write_to_file:
			opened_file.writelines(["Resultant encoding: "])
			write_list(num_to_tag(encoding), opened_file, ignore_t=True)
			opened_file.write("\n")

		return True

	return False

def check_punc(index, index_2, combined_s, encoding, tokenized_arg, diff_list, file_name, opened_file, turn_num, write_to_file):
	"""
	Modifies the encoding accordingly if it's a punctuation edge case (missing punctuation, "I" tag used instead of "O" tag)

	index: the index of the encoding
	index_2: the current diff list index
	combined_s: the gold segmentation of the turns that have been unsegmented
	encoding: the encoded turn
	tokenized_arg: the preprocessed, tokenized turn
	diff_list: the python diff between the encoding and tokenized_arg
	file_name: the file name of the transcript
	opened_file: the edge case file handle written to
	turn_num: the turn number in the transcript
	write_to_file: a flag that determines if the edge case file is written to

	return: None
	"""
	end_punc_list = [",", ".", "?", "!"]
	if diff_list[index_2][2] in end_punc_list:
		if write_to_file:
			line = ["File: " + file_name + "\n", "Turn Num: " + turn_num + "\n", "Error: Missing Punctuation \n"]
			opened_file.writelines(line)
			opened_file.writelines(["Raw text: "])
			write_list(tokenized_arg, opened_file)
			opened_file.writelines(["Segmented text: "])
			write_list(combined_s, opened_file)
			opened_file.write("Fix: Inserted 2 from Raw Text T" + str(index_2) + "\n")

		encoding.insert(index, 2)

		if write_to_file:
			opened_file.writelines(["Resultant encoding: "])
			write_list(num_to_tag(encoding), opened_file, ignore_t=True)
			opened_file.write("\n")

	else:
		encoding.insert(index, 1)

def insert_Os(tokenized_arg_list, segments_list, encodings_y, file_name, opened_file, out, write_to_file=True):
	"""
	Insert "O" tags into the preprocessed tokenized turns and the encoded turns, has option to write edge case
	instances to a file

	tokenized_arg_list: the preprocessed, tokenized turns
	segment_list: the gold segmentation of the turns (no encoding)
	encodings_y: the encoded turns
	file_name: the file name of the edge case file
	opened_file: the edge case file handle written to
	out: the raw parsed transcript
	write_to_file: a flag that determines if the edge case file is written to

	return: None
	"""

	for i in range(len(tokenized_arg_list)):
		combined_s = []
		list(map(combined_s.extend, segments_list[i]))
		skip_check = False

		if(len(tokenized_arg_list[i]) != len(combined_s)):
			# print("Raw Text  : ", end = '')
			# print(tokenized_arg_list[i])
			# print("Seg Before: ", end = '')
			# print(combined_s)
			diff_list = list(d1.Differ().compare(combined_s, tokenized_arg_list[i]))

			if check_space(combined_s, encodings_y[i], tokenized_arg_list[i], diff_list, file_name, opened_file, out["turnList"][i]["TurnNum"], write_to_file):
				continue
			
			if check_copy(combined_s, encodings_y[i], tokenized_arg_list[i], diff_list, file_name, opened_file, out["turnList"][i]["TurnNum"], write_to_file):
				continue

			else:
				index = 0
				for j in range(len(diff_list)):
					if diff_list[j][0] == "+":
						# encodings_y[i].insert(index, 1)
						check_punc(index, j, combined_s, encodings_y[i], tokenized_arg_list[i], diff_list, file_name, opened_file, out["turnList"][i]["TurnNum"], write_to_file)
					elif diff_list[j][0] == "-":
						del encodings_y[i][index]
						index -= 1
					index += 1

		# Checking for bigram edge cases from O-insertions
		# if len(encodings_y[i]) > 1:
		# 	for k in range(len(encodings_y[i]) - 1):
		# 		if encodings_y[i][k] == 1 and encodings_y[i][k + 1] == 2:
		# 			print(file_name)
		# 			print(out["turnList"][i]["TurnNum"])
		# 			print(segments_list[i])
		# 			print(encodings_y[i])
			# if encodings_y[i][0] == 0 and encodings_y[i][1] == 0:
			# 	print(file_name)
			# 	print(out["turnList"][i]["TurnNum"])
			# 	print(segments_list[i])

			# print("Seg After:  ", end = '')	
			# print(combined_s)
			# print(encodings_y[i])

def preprocess_chunk(chunk, file_name=""):
	"""
	Converts turn to unicode, tokenizes it, and compresses the parens
	instances

	chunk: the raw turn
	file_name: the file name of the transcript

	return: None
	"""

	parser = CoreNLPParser(url='http://localhost:9000')

	chunk = unidecode(chunk)
	chunk = list(parser.tokenize(chunk))

	try:
		compress_parens(chunk, "{", "}", file_name)
	except:
		pass
	try:
		compress_parens(chunk, "[", "]", file_name)
	except:
		pass
	try:
		compress_parens(chunk, "(", ")", file_name)
	except:
		pass

	# print(chunk)
	# chunk = chunk.replace('\\xa0', "")
	# chunk = chunk.replace('{overlapping}', "")
	# chunk = chunk.replace('xx', "")
	# chunk = re.sub(r'\[.*?\]', "", chunk)

	# # hyphen token
	# chunk = chunk.replace("-", "")
	
	# # aux verb
	# aux_verbs = ["might", "may", "must", "shall", "have", "has", "having"] # incomplete list
	# for verb in aux_verbs:
	# 	chunk = chunk.replace(verb, "")

 #    # specific de-contractions
	# chunk = re.sub(r"won\'t", "will not", chunk)
	# chunk = re.sub(r"can\'t", "can not", chunk)

	# general de-contractions
	# chunk = re.sub(r"n\'t", " not", chunk)
	# chunk = re.sub(r"\'re", " are", chunk)
	# chunk = re.sub(r"\'s", " is", chunk)
	# chunk = re.sub(r"\'d", " would", chunk)
	# chunk = re.sub(r"\'ll", " will", chunk)
	# chunk = re.sub(r"\'t", " not", chunk)
	# chunk = re.sub(r"\'ve", " have", chunk)
	# chunk = re.sub(r"\'m", " am", chunk)
	return chunk

def compress_parens(x, left, right, file_name):
	"""
	Compresses parens instances depending on type provided,
	including curly braces, brackets, and parenthesis

	x: the preprocessed turn
	left: the left parens
	right: the right parens
	file_name: the file name of the transcript

	return: None
	"""
	global curly_list
	global brack_list
	global paren_list
	global file_dict

	while x.count(left) > 0:
		# print(x)
		j = x.index(left)
		closeIdx = x.index(right, j)
		temp = ' '.join(x[j:closeIdx+1])
		x[j] = temp
		del x[j+1:closeIdx+1]

		# Uncomment if you want parens to be removed
		# del x[j]

		# For printing out parens instances
		# if left == "{":
		# 	# curly_list.append(x[j])
		# elif left == "[":
		# 	# brack_list.append(x[j])
		# else:
		# 	# paren_list.append(x[j])

		# print(x)
		# print("\n")

		# Keep count in file
		# if file_name in file_dict.keys():
		# 	file_dict[file_name] += 1
		# else:
		# 	file_dict[file_name] = 0

		# Print accumulating counts
		# print("Num Curly: " + str(len(curly_list)))
		# print("Num Brackets: " + str(len(brack_list)))
		# print("Num Parentheses: " + str(len(paren_list)))
		# print(file_dict)

		# Print accumulating lists
		# print(curly_list)
		# print(brack_list)
		# print(paren_list)
		# print("\n")




def check_lengths(x, y, z):
	for i in range(len(x)):
		if len(x[i]) != len(z[i]):
			# print("Arg: ", end="")
			# print(x[i])
			# print("Seg: ", end="")
			# print(y[i])
			# print("Enc: ", end="")
			# print(z[i])
			# print("\n")
			return False

	return True
