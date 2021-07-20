from parse_transcript import *
# from transcriptValidator import *
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

################## Rule-based Segmentation ##################
# 1. Label each token of argument with pos and dependencies
# 2. Create a ruleset for when a segment might begin or end
# 3. Generate N choose 2 - 2 * (N/2 choose 2) different trios of segments for each argument
# 4. Compare segments created from hand picked rules to true segments
# 5. Pick from 0 through n x n as rule and apply to unseen data 

################################# Old Rule Set #######################################
rule_set = {"rule_1": {'left': ['nsubj'], 'right': ['.', '?', '!']}, \
			"rule_2": {'left': ['WP', 'WP$'], 'right': ['.', '?', '!']}, \
			"rule_3": {'left': ['WP', 'WP$'], 'right': ['.', '?', '!', 'CC']}, \
			"rule_4": {'left': ['nsubj', 'WP', 'WP$'], 'right': ['CC']}, \
			"rule_5": {'left': ['CC'], 'right': ['.', '?', '!']}, \
			"rule_6": {'left': ['Um', 'Uh', 'Like', 'CC'], 'right': ['.', '?', '!']}}
			# "rule_8": {'left': [], 'right': []}, \
			# }

######################### Modified Rule Set (In Progress) ###########################
rule_set_2 = {1: ['nsubj'], 2: ['.', '?', '!'],
			  3: ['WP', 'WP$'], 4: ['.', '?', '!', 'CC'],
			  5: ['nsubj', 'WP', 'WP$'], 6: ['CC'],
			  7: ['Um', 'Uh', 'Like', 'CC'], 8: ['.']}

# N = number of left or right rules
# N_pairs = N choose 2 - 2 * (N/2 choose 2)
# With N = 8, N_pairs = 16

pairs = list(itertools.combinations(list(rule_set_2.keys()), 2))
pairs = [pair for pair in pairs if (pair[0] + pair[1]) % 2 != 0]
######################################################################################		

# To be used globally
curly_list = []
brack_list = []
paren_list = []
file_dict = {}

def get_gold_stats(file_dir):
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
	for x in range(len(encodings)):
		if collections.Counter(encodings[x])[0] != counts_x[x]:
			print("File: " + str(file_name))
			print("Turn Num: " + str(x + 1))
			print("Raw Text: ", end="")
			print(arg_list[x])
			print("Encoding: ", end="")
			print(num_to_tag(encodings[x]))
			print("\n")

			# tokenized_sentence = preprocess_chunk(arg_list[x])
			# count = 0
			# segment = []
			# for i in range(len(encodings[x])):	
			# 	if i == 0:
			# 		segment = []
			# 		segment.append(tokenized_sentence[i])
			# 		count += 1
			# 	elif encodings[x] == 2:
			# 		segment.append(tokenized_sentence[i])
			# 	elif encodings[x] == 0:
			# 		print("Segment " + str(count) + ": ", end = '')
			# 		print(segment)
			# 		segment = []
			# 		segment.append(tokenized_sentence[i])
			# 		count += 1
			# 	if i == len(encodings[x]) - 1:
			# 		print("Segment " + str(count) + ": ", end = '')
			# 		print(segment)
			# print("\n")			

# TO DO
def replace_lower_postprocess(encoding, counts_x, index, lower_bound):
	if counts_x[index] < lower_bound:
		# print("Before")
		# print(num_to_tag(encoding))
		encoding[0] = 0
		# print("After")
		# print(num_to_tag(encoding))
		# print("\n")

# TO DO
def replace_upper_postprocess(encoding, counts_x, index, upper_bound):	
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

# TO DO
def bigram_postprocess(encoding):
	for i in range(len(encoding) - 1):
		if num_to_tag(encoding)[i] + num_to_tag(encoding)[i + 1] ==  "BB":
			# print("Before")
			# print(num_to_tag(encoding))
			encoding[i + 1] = 2
			# print("After")
			# print(num_to_tag(encoding))
			# print("\n")

def num_to_tag(encoding):
	return_encoding = copy.deepcopy(encoding)
	encoding_map = {0:"B", 2:"I", 1:"O"}
	for i in range(len(encoding)): 
		return_encoding[i] = encoding_map[float(encoding[i])]

	return return_encoding

def tag_to_num(encoding):
	return_encoding = copy.deepcopy(encoding)
	encoding_map = {"B":0, "I":2, "O":1}
	for i in range(len(encoding)): 
		return_encoding[i] = encoding_map[encoding[i]]
	
	return return_encoding

def verify(segment_mat, token_mat, include_zero=False):
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
	# num_segments_list = []
	# # enc_list = []
	# count = 0
	# for x in encodings:
	# 	count = collections.Counter(x)[0]

	# 	if count > 4:
	# 		count = 4

	# 	num_segments_list.append(count)

	num_segments_list = [collections.Counter(x)[0] if collections.Counter(x)[0] < 4 else 4 for x in encodings]

	return num_segments_list

def segment_counter_3(encodings):
	# num_segments_list = []
	# # enc_list = []
	# count = 0
	# for x in encodings:
	# 	count = collections.Counter(x)[0]

	# 	if count == 0:
	# 		count = 1

	# 	num_segments_list.append(count)

	num_segments_list = [collections.Counter(x)[0] if collections.Counter(x)[0] != 0 else 1 for x in encodings]

	return num_segments_list

def segment_counter_2(encodings):
	num_segments_list = [collections.Counter(x)[0] for x in encodings]

	return num_segments_list

def segment_counter_1(encodings):
	num_segments_list = []
	# enc_list = []
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
	num_segments_list = []
	count = 0
	for x in collab:
		for y in x["Argumentation"]:
			if y != '' and y != "None":
				count += 1
		num_segments_list.append(count)
		count = 0
	# print(num_segments_list)
	# num_segments_list = [len(x["Argumentation"][x["Argumentation"] != '']) if len(x["Argumentation"]) > 1 else 1 for x in collab]
	return num_segments_list

def print_segments(encoding_2, encoding_1, tokenized_sentence, opened_file, write_to_file=False):
	# print("\n")
	key_words_y = []
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
			# stack = []
			# stack.append(test_y_encodings[j][i])
			# check_append(segment, tokenized_sentence[i])
		elif encoding_1[i] == 2:
			segment.append(tokenized_sentence[i])
			# check_append(segment, tokenized_sentence[i])
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
			# stack = []
			# stack.append(test_y_encodings[j][i])
			# check_append(segment, tokenized_sentence[i])
			# true_segment_count += 1

		if i == len(encoding_1) - 1:
			# print("Segment " + str(count) + ": ", end = '')
			# print(segment)

			if write_to_file:
				opened_file.writelines(["Segment " + str(count) + ": "])
				write_list(segment, opened_file, ignore_t=True)

			# true_segment_count += 1

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

def decode_x(test_x_encodings, test_y_encodings, test_arg_list, segment_list, key_words_x, key_words_y, opened_file=None, write_to_file=False, file_name=None):
	##################### Decode test set and show the segments created  ##########################
	# print(test_x_encodings[1])
	test = []
	segment = []
	true_segment_count = 0
	pred_segment_count = 0
	desired_correct_segments = {1:0, 2:0, 3:0, 4:0}
	num_correct_segments = {1:0, 2:0, 3:0, 4:0}
	temp_dict = {"11":0, "12": 0, "13": 0, "14": 0, "21":0, "22": 0, "23": 0, "24": 0, "31":0, "32": 0, "33": 0, "34": 0, "41":0, "42": 0, "43": 0, "44": 0}
	sub_segment_list = []

	# count_x = 0
	# count_y = 0
	# for i in range(len(test_y_encodings)):
	# 	count_y = collections.Counter(test_y_encodings[i])[0]
	# 	count_x = collections.Counter(test_x_encodings[i])[0]

	# 	if count_y == 0:
	# 		count_y = 1
	# 	if count_x == 0:
	# 		count_x = 1


	# 	if count_x > 4:
	# 		count_x = 4
	# 	if count_y > 4:
	# 		count_y = 4

	# 	counts_x.append(count_x)
	# 	counts_y.append(count_y)
	# print(len(counts_x))

		# desired_correct_segments[count] += 1
		# num_correct_segments[count2] += 1

	# print(desired_correct_segments)
	# print(num_correct_segments)

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
		# count_x, count_y = gold_counter(test_x_encodings[j], test_y_encodings[j])
		# count_x, count_y = segment_count_2(test_x_encodings[j], test_y_encodings[j])

		# counts_x.append(count_x)
		# counts_y.append(count_y)




		# segment_list.append(sub_segment_list)
		# sub_segment_list = []

	# 	# if true_segment_count == 5:
	# 	# 	if true_segment_count == pred_segment_count:
	# 	# 		num_correct_segments[5] += 1
	# 	# 		desired_correct_segments[5] += 1
	# 	# 	else:
	# 	# 		# num_correct_segments[pred_segment_count] += 1
	# 	# 		desired_correct_segments[5] += 1

	# 	# if true_segment_count == 4:
	# 	# 	if true_segment_count == pred_segment_count:
	# 	# 		num_correct_segments[4] += 1
	# 	# 		desired_correct_segments[4] += 1
	# 	# 	else:
	# 	# 		# num_correct_segments[pred_segment_count] += 1
	# 	# 		desired_correct_segments[4] += 1


	# 	if true_segment_count == 3:
	# 		if true_segment_count == pred_segment_count:
	# 			num_correct_segments[3] += 1
	# 			desired_correct_segments[3] += 1
	# 		else:
	# 			# num_correct_segments[pred_segment_count] += 1
	# 			desired_correct_segments[3] += 1

	# 	if true_segment_count == 2:
	# 		if true_segment_count == pred_segment_count:
	# 			num_correct_segments[2] += 1
	# 			desired_correct_segments[2] += 1
	# 		else:
	# 			# num_correct_segments[pred_segment_count] += 1
	# 			desired_correct_segments[2] += 1

	# 	if true_segment_count == 1:
	# 		if true_segment_count == pred_segment_count:
	# 			num_correct_segments[1] += 1
	# 			desired_correct_segments[1] += 1
	# 		else:
	# 			# num_correct_segments[pred_segment_count] += 1
	# 			desired_correct_segments[1] += 1

	# 	if pred_segment_count >= 4:
	# 		pred_segment_count = 4

	# 	if true_segment_count >= 4:
	# 		print("ERROR: " + str(file_name))
	# 		print(test_arg_list[j])

	# 	counts_y.append(true_segment_count)
	# 	counts_x.append(pred_segment_count)

	# 	# temp_dict[str(true_segment_count) + str(pred_segment_count)] += 1

	# 	true_segment_count = 0
	# 	pred_segment_count = 0

	return [file_name, num_correct_segments, desired_correct_segments]

def check_append(segment, token):
	if token != "<pad>":
		segment.append(token)


def extend_list(shorter_list, longer_list, index, option):
	##################### Temporary fix off by 1 error by extending ############################
	long_len = len(longer_list)
	short_len = len(shorter_list)
	missing_len = long_len - short_len

	# Zero padding at the end	
	# shorter_list.extend([0] * (missing_len))

	# One padding in the middle
	for i in range(missing_len):
		shorter_list.insert(1, 2)

	return shorter_list

def list_to_arrays_and_pad(encodings_x, encodings_y):
	################# Convert lists to arrays and pad with 0s (for now) ##########################
	total_samples = len(encodings_y)
	# print(encodings_x)

	# Should probably have made the encodings np arrays in the first place 
	for x in range(total_samples):
		encodings_y[x] = np.asarray(extend_list(encodings_y[x], encodings_x[list(encodings_x.keys())[0]][x], x, 0))
	for key in encodings_x.keys():
		for x in range(total_samples):
			encodings_x[key][x] = np.asarray(extend_list(encodings_x[key][x], encodings_y[x], x, 1))
	return encodings_x, encodings_y

def class_report_segment_level(counts_x, counts_y, path, write_to_csv=True):
	target_names = ["1", "2", "3"]
	target_names_2 = ["1", "2", "3", "4"]
	target_names_3 = ["1", "2", "3", "4", "5"]
	# target_names = ["0", "1", "2", "3", "4", "5",]


	prec_dict = {"Precision_for_1":0, "Precision_for_2":0, "Precision_for_3":0}
	f_dict = {"F_for_1":0, "F_for_2":0, "F_for_3":0}
	recall_dict = {"Recall_for_1":0, "Recall_for_2":0, "Recall_for_3":0}

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

	return prec_dict, recall_dict, f_dict	

def conf_mat_segment_level(counts_x, counts_y, path, write_to_csv=True):
	# print(counts_y)
	# print(counts_x)
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
	############ Compute precision and recall of each rule (only for train set analysis) #################
	total_samples = len(encodings_y)
	# print(len(encodings_y))
	# print(len(encodings_x))
	prec_score_0 = 0
	recall_score_0 = 0
	f_score_0 = 0

	prec_score_2 = 0
	recall_score_2 = 0
	f_score_2 = 0

	prec_score_1 = 0
	recall_score_1 = 0
	f_score_1 = 0

	target_names = ["0-beginning", "1-padding", "2-inner"]
	# target_names_2 = ["2-inner", "1-padding"]
	# target_names_3 = ["2-inner"]

	# target_names = ["0-beginning", "2-inner"]


	# highest_precision = 0
	# highest_recall = 0

	prec_dict = {"Precision_for_0":0, "Precision_for_2":0, "Precision_for_1":0}
	f_dict = {"F_for_0":0, "F_for_2":0, "F_for_1":0}
	recall_dict = {"Recall_for_0":0, "Recall_for_2":0, "Recall_for_1":0}

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

	return prec_dict, recall_dict, f_dict


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

	# print(encodings_y)


	# print("enc: " + str(len(encodings_y)))



		# if turn_segments_list[x]:
		# 	doc = nlp(preprocess_chunk(turn_segments_list[x]))
		# 	for i, sent in enumerate(doc.sentences):
		# 	    for word in sent.words:
		# 	        # print("{:12s}\t{:12s}\t{:6s}\t{:d}\t{:12s}".format(\
		# 	        #       word.text, word.lemma, word.pos, word.governor, word.dependency_relation))
		# 	        sub_pos_list.append(word.pos)
		# 	pos_list.append(sub_pos_list)
		# 	sub_pos_list = []
		# else:
		# 	pos_list.append("")
	# print("Y")
	# print(len(turn_segments_list))
	# print(len(pos_list))
	# print(int((len(pos_list))/rows))
	# print(pos_list)
	# print(turn_segments_list)

	# for x in range(int((len(pos_list))/rows)):
	# 	for y in range(segments):
	# 		if pos_list[x * rows + y]:
	# 			temp_list = pos_list[x * rows + y]
	# 			# print((temp_list))
	# 			for z in range(len(temp_list)):
	# 				if z == 0:
	# 					temp_list[z] = 0 # + y/10
	# 				else:
	# 					temp_list[z] = 2 # + y/10
	# 			token_list = token_list + temp_list
	# 	# if only_multi:
	# 	# 	if check_multi(token_list):
	# 	# 		# print(len(token_list))
	# 	# 		encodings_y.append(token_list)
	# 	# 	else:
	# 	# 		remove_list.append(x)
	# 	# else:
	# 	encodings_y.append(token_list)		
	# 	token_list = []
	# print(encodings_y)
	# Save as a list so you don't have to start up CoreNLP server
	# with open(os.path.join(cv_dir, y_list_fn), 'wb') as filehandle:
	# 	pickle.dump(encodings_y, filehandle)

	# with open(os.path.join(cv_dir, y_list_fn), 'rb') as filehandle:
	#     # read the data as binary data stream
	#     encodings_y = pickle.load(filehandle)
	# print("Encodings")
	# print(encodings_y[0:50])
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


def encode_x(collab, train_or_test=0, best_rule=0):
	#################################### Read arguments, pos, dep lists ######################################
	make_arg_list(collab)
	with open(os.path.join(cv_dir, arg_list_fn), 'rb') as filehandle:
	   	arg_list = pickle.load(filehandle)

	apply_stan_models(arg_list)
	with open(os.path.join(cv_dir, pos_list_fn), 'rb') as filehandle:
	    pos_list = pickle.load(filehandle)

	with open(os.path.join(cv_dir, dep_list_fn), 'rb') as filehandle:
	    dep_list = pickle.load(filehandle)


	# Get encodings with these lists
	if train_or_test == 0:
		encodings_x = apply_rules_train(pos_list, dep_list, arg_list)
		return encodings_x
	else:
		test_encodings = apply_rules_test(pos_list, dep_list, arg_list, best_rule)
		return test_encodings

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
	# Missing space edge case, save where this happens in an external file with transcript, turn number, raw text, segmented, resultant encoding, edge case type
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
	# Punctuation edge case (inner tag instead of outside tag)
	# print("marker")
	# print(diff_list[2])
	# print(len(diff_list[2]))
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

		# if left == "{":
		# 	# curly_list.append(x[j])
		# 	del x[j]
		# elif left == "[":
		# 	# brack_list.append(x[j])
		# 	# Removal of bracket instances
		# 	del x[j]

		# else:
		# 	# paren_list.append(x[j])
		# 	# Removal of parenthesis instances
		# 	del x[j]

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

############################################################### OLD STUFF ##################################################################################


def apply_stan_models(arg_list):
	################# Apply Stanford NLP models for POS and dependency relation ######################
	pos_list = []
	sub_pos_list = []
	dep_list = []
	sub_dep_list = []
	'''
	-apostrophes
	-quotes
	-
	'''
	# print(arg_list)
	# stanfordnlp.download('en')
	nlp = stanfordnlp.Pipeline()
	for x in range(len(arg_list)):
		doc = nlp(arg_list[x])
		for i, sent in enumerate(doc.sentences):
		    for word in sent.words:
		        # print("{:12s}\t{:12s}\t{:6s}\t{:d}\t{:12s}".format(\
		        #       word.text, word.lemma, word.pos, word.governor, word.dependency_relation))
		        sub_pos_list.append(word.pos)
		        sub_dep_list.append(word.dependency_relation)
		# print(len(sub_pos_list))
		pos_list.append(sub_pos_list)
		dep_list.append(sub_dep_list)
		sub_pos_list = []
		sub_dep_list = []
		
	# print("X")
	# print(arg_list[2])
	# print(pos_list[len(pos_list) - 3])
	# print((pos_list[25]))
	# Save as a list to save time
	# POS List
	# with open(os.path.join(cv_dir, pos_list_fn), 'wb') as filehandle:
	# 	pickle.dump(pos_list, filehandle)

	# # # Dependency List
	# with open(os.path.join(cv_dir, dep_list_fn), 'wb') as filehandle:
	# 	pickle.dump(dep_list, filehandle)

	return pos_list, dep_list


def apply_rules_test(pos_list, dep_list, arg_list, rule):
	##################### Encode test set based on the rule provided ##############################
	rule_encoding = []
	sub_rule_encoding = []

	test_encodings = {rule: []}
	rule_encoding = {rule: []}
	
	sub_rule_encoding = 0

	# 
	for x in range(len(dep_list)):
		for y in range(len(dep_list[x])):
			# if pos_list[x][y] not in ['-LRB-', '-RRB-', '``', '\'\'']:
				if dep_list[x][y] in rule_set_2[int(rule[6])] \
					or pos_list[x][y] in rule_set_2[int(rule[6])] \
					or arg_list[x][y] in rule_set_2[int(rule[6])] \
					and arg_list[x][y][0].isupper():
					sub_rule_encoding = 0
				elif dep_list[x][y] in rule_set_2[int(rule[9])] \
					or pos_list[x][y] in rule_set_2[int(rule[9])] \
					or arg_list[x][y] in rule_set_2[int(rule[9])]:
					sub_rule_encoding = 2
				else:
					sub_rule_encoding = 2
				rule_encoding[rule].append(sub_rule_encoding)
		rule_encoding[rule][len(rule_encoding[rule]) - 1] = 2
		test_encodings[rule].append(rule_encoding[rule])
		rule_encoding[rule] = []


	return test_encodings

def apply_rules_train(pos_list, dep_list, arg_list):
	##################### Encode train set based on the rule provided ##############################
	encodings_x = {"Keys: " + str(pair[0]) + ", " + str(pair[1]) + "|" \
					+ "|".join(["Left: " + ", ".join(rule_set_2[pair[0]]), "Right: " \
					+ ", ".join(rule_set_2[pair[1]])]):[] for pair in pairs}
	
	rule_encoding = {"Keys: " + str(pair[0]) + ", " + str(pair[1]) + "|" \
					+ "|".join(["Left: " + ", ".join(rule_set_2[pair[0]]), "Right: " \
					+ ", ".join(rule_set_2[pair[1]])]):[] for pair in pairs}	
	# print(encodings_x)
	rule = ""
	for x in range(len(dep_list)):
		# print(len(dep_list[x]))
		for y in range(len(dep_list[x])):
			for pair in pairs:					
				rule = "Keys: " + str(pair[0]) + ", " + str(pair[1]) + "|" \
				+ "|".join(["Left: " + ", ".join(rule_set_2[pair[0]]), "Right: " \
				+ ", ".join(rule_set_2[pair[1]])])
				if dep_list[x][y] in rule_set_2[pair[0]] \
					or pos_list[x][y] in rule_set_2[pair[0]] \
					or arg_list[x][y] in rule_set_2[pair[0]] \
					and arg_list[x][y][0].isupper():
					sub_rule_encoding = 0
				elif dep_list[x][y] in rule_set_2[pair[1]] \
					or pos_list[x][y] in rule_set_2[pair[1]] \
					or arg_list[x][y] in rule_set_2[pair[1]]:
					sub_rule_encoding = 2
				else:
					sub_rule_encoding = 2
				rule_encoding[rule].append(sub_rule_encoding)
				rule = ""

		for pair in pairs:
			rule = "Keys: " + str(pair[0]) + ", " + str(pair[1]) + "|" \
					+ "|".join(["Left: " + ", ".join(rule_set_2[pair[0]]), "Right: " \
					+ ", ".join(rule_set_2[pair[1]])])
			encodings_x[rule].append(rule_encoding[rule])
			rule_encoding[rule] = []

	# print(encodings_x)
	return encodings_x


def make_comparisons_class_report(encodings_x, encodings_y, results_fn):
	############ Compute precision and recall of each rule (only for train set analysis) #################
	total_samples = len(encodings_y)
	encodings_x, encodings_y = list_to_arrays_and_pad(encodings_x, encodings_y)

	prec_dict = {"Keys: " + str(pair[0]) + ", " + str(pair[1]) + "|" \
					+ "|".join(["Left: " + ", ".join(rule_set_2[pair[0]]), "Right: " \
					+ ", ".join(rule_set_2[pair[1]])]):{"Precision for 0":0, "Precision for 2": 0} for pair in pairs}
	recall_dict = {"Keys: " + str(pair[0]) + ", " + str(pair[1]) + "|" \
					+ "|".join(["Left: " + ", ".join(rule_set_2[pair[0]]), "Right: " \
					+ ", ".join(rule_set_2[pair[1]])]):{"Recall for 0":0, "Recall for 2": 0}  for pair in pairs}

	prec_score_0 = 0
	recall_score_0 = 0

	prec_score_2 = 0
	recall_score_2 = 0
	target_names = ["0-beginning", "2-inner"]

	highest_precision = 0
	highest_recall = 0

	# Iterate through all the rules, compute F-score average for each rule for micro and macro at the same time
	for key in encodings_x.keys():
		for x in range(total_samples):
			report = classification_report(encodings_y[x], encodings_x[key][x], target_names=target_names)
			lines = report.split('\n')

			row_data = lines[2].split('      ')
			prec_score_0 += float(row_data[1])
			recall_score_0 += float(row_data[2])

			row_data = lines[3].split('      ')
			prec_score_2 += float(row_data[1])
			recall_score_2 += float(row_data[2])

		prec_dict[key]["Precision_for_0"] = prec_score_0/total_samples
		prec_dict[key]["Precision_for_2"] = prec_score_2/total_samples
		recall_dict[key]["Recall_for_0"] = recall_score_0/total_samples
		recall_dict[key]["Recall_for_2"] = recall_score_2/total_samples
		prec_score_0 = 0
		recall_score_0 = 0
		prec_score_2 = 0
		recall_score_2 = 0	

	# Write to full results in csv format, including the average + highest precision and average + highest recall
	sum_list_0 = []
	sum_list_2 = []
	with open(os.path.join(cv_dir, results_fn), 'a', newline='') as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow(["\n"])            
		csv_writer.writerow(["Precision"])
		csv_writer.writerow(["Key", "0-beginning", "2-end"])
		for key in prec_dict.keys():
			csv_writer.writerow([key, prec_dict[key]["Precision_for_0"], prec_dict[key]["Precision_for_2"]])
			sum_list_0.append(prec_dict[key]["Precision_for_0"])
			sum_list_2.append(prec_dict[key]["Precision_for_2"])
		csv_writer.writerow(["Average", sum(sum_list_0)/len(sum_list_0), sum(sum_list_2)/len(sum_list_2)])
		csv_writer.writerow(["Best", max(sum_list_0), max(sum_list_2)])

	sum_list_0 = []
	sum_list_2 = []
	with open(os.path.join(cv_dir, results_fn), 'a', newline='') as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow(["\n"])           
		csv_writer.writerow(["Recall"])
		csv_writer.writerow(["Key", "0-beginning", "2-end"])
		for key in prec_dict.keys():
			csv_writer.writerow([key, recall_dict[key]["Recall_for_0"], recall_dict[key]["Recall_for_2"]])
			sum_list_0.append(recall_dict[key]["Recall_for_0"])
			sum_list_2.append(recall_dict[key]["Recall_for_2"])
		csv_writer.writerow(["Average", sum(sum_list_0)/len(sum_list_0), sum(sum_list_2)/len(sum_list_2)])
		csv_writer.writerow(["Best", max(sum_list_0), max(sum_list_2)])	

def make_comparisons_f_score(encodings_x, encodings_y, results_fn, train_or_test):
	################# Compute F score average for micro and macro of each rule ####################
	total_samples = len(encodings_y)
	encodings_x, encodings_y = list_to_arrays_and_pad(encodings_x, encodings_y)

	f_dict = {}
	f_score = 0
	max_f_score = 0;
	best_rule = ""

	f_dict_macro = {}
	f_score_macro = 0
	max_f_score_macro = 0;
	best_rule_macro = ""

	# Iterate through all the rules, compute F-score average for each rule for micro and macro at the same time
	for key in encodings_x.keys():
		for x in range(total_samples):
			f_score += f1_score(encodings_y[x], encodings_x[key][x], average='micro')
			f_score_macro += f1_score(encodings_y[x], encodings_x[key][x], average='macro')

		f_dict[key] = f_score/total_samples
		f_dict_macro[key] = f_score_macro/total_samples

		if f_dict[key] > max_f_score:
			max_f_score = f_dict[key]
			best_rule = key
		if f_dict_macro[key] > max_f_score_macro:
			max_f_score_macro = f_dict_macro[key]
			best_rule_macro = key

		f_score = 0
		f_score_macro = 0

	# Write to full results in csv format, including the best rule/highest F-score
	# if write_enable == 1:
	# 	with open(os.path.join(cv_dir, results_fn), 'a', newline='') as csvfile:
	# 		csv_writer = csv.writer(csvfile)           
	# 		csv_writer.writerow(["", "Micro", "Macro"])
	# 		csv_writer.writerow(["Key", "F-score", "F-score"])
	# 		for key in f_dict.keys():
	# 			csv_writer.writerow([key, f_dict[key], f_dict_macro[key]])
	# 		csv_writer.writerow(["Best F-Score", max_f_score, max_f_score_macro])
	# 		csv_writer.writerow(["Best Rule", best_rule, best_rule_macro])

	return best_rule, f_dict, best_rule_macro, f_dict_macro

if __name__ == "__main__":
	#################################### Get necessary raw dicts and lists ####################################
	# file_path = "../Data/EAGER/T127.EAGER.1.Heartdark.Final.xlsx"
	file_path = "../Data/EAGER/T124.EAGER.2.Night.Final.xlsx"
	out = convert_transcript(file_path)
	collab = to_TurnSegmentation(out)
	num_turns = len(collab)
	num_rows = 5
	turn_segments = get_argument_segments(file_path).tolist()

	# Handling results.txt
	# write_enable = 1
	results_fn_1 = "results_micro.txt"
	results_fn_2 = "results_macro.txt"
	prec_fn = "prec.csv"
	recall_fn = "recall.csv"

	full_results_fn = "results.csv"

	cv_dir = 'cv_trial_1_all'
	# if write_enable == 1:
	# 	if os.path.exists(os.path.join(cv_dir, results_fn_1)):
	# 	  os.remove(os.path.join(cv_dir, results_fn_1))
	# 	with open(os.path.join(cv_dir, full_results_fn), 'w') as csvfile:
	# 		csv_writer = csv.writer(csvfile) 
	# 		csv_writer.writerow(["F-Score, Precision, and Recall for CV 1"])
	#################################### Skip right to cross validatation ####################################
	# 3 fold cross validation, k = number of folds, n = k - 1
	k = 3
	kf = KFold(n_splits=k, random_state=None, shuffle=False)

	train_values = []
	test_values = []
	sets = [x for x in kf.split(collab)]
	results = []


	for x in range(k):
		# Train portion
		# if write_enable == 1:
		# 	with open(os.path.join(cv_dir, full_results_fn), 'a') as csvfile:
		# 		csv_writer = csv.writer(csvfile) 
		# 		csv_writer.writerow(["Train"])
			# with open(os.path.join(cv_dir, results_fn_2), 'a') as file:
			# 	file.write("\nTrain " + str(x + 1) + "\n");
			# with open(os.path.join(cv_dir, results_fn_3), 'a') as file:
			# 	file.write("\nTrain " + str(x + 1) + "\n");

		arg_list_fn = 'arg_list_cv_train' + str(x + 1) + '.data'
		pos_list_fn = 'pos_list_cv_train' + str(x + 1) + '.data'
		dep_list_fn = 'dep_list_cv_train' + str(x + 1) + '.data'
		y_list_fn = 'encodings_y_cv_train' + str(x + 1) + '.data'

		next_set = sets[x]
		train_x_indicies = next_set[0].tolist()
		test_x_indicies = next_set[1].tolist()
		train_x_values = [collab[index] for index in train_x_indicies]
		test_x_values = [collab[index] for index in test_x_indicies]

		train_y_indicies = [next_set[0][0]*num_rows + x for x in range(len(train_x_indicies)*num_rows)]
		test_y_indicies = [next_set[1][0]*num_rows + x for x in range(len(test_x_indicies)*num_rows)]
		train_y_values = [turn_segments[index] for index in train_y_indicies]
		test_y_values = [turn_segments[index] for index in test_y_indicies]

		train_x_encodings = encode_x(train_x_values)
		train_y_encodings = encode_y(train_y_values)

		best_rule_micro, scores_micro, best_rule_macro, scores_macro  = make_comparisons_f_score(train_x_encodings, train_y_encodings, full_results_fn, 0)


		# # Test portion
		# if write_enable == 1:
		# 	with open(os.path.join(cv_dir, full_results_fn), 'a') as csvfile:
		# 		csv_writer = csv.writer(csvfile) 
		# 		csv_writer.writerow(["\n"])
		# 		csv_writer.writerow(["Test"])
		arg_list_fn = 'arg_list_cv_test' + str(x + 1) + '.data'
		pos_list_fn = 'pos_list_cv_test' + str(x + 1) + '.data'
		dep_list_fn = 'dep_list_cv_test' + str(x + 1) + '.data'
		y_list_fn = 'encodings_y_cv_test' + str(x + 1) + '.data'

		test_x_encodings_micro = encode_x(test_x_values, 1, best_rule_micro)
		test_x_encodings_macro = encode_x(test_x_values, 1, best_rule_macro)
		test_y_encodings = encode_y(test_y_values)
		make_comparisons_f_score(test_x_encodings_micro, test_y_encodings, full_results_fn, 1)

		# Get Precision and Recall too
		make_comparisons_class_report(train_x_encodings, train_y_encodings, full_results_fn)

	# 	# Decode predicted segments

		# with open(os.path.join(cv_dir, "test_encoding_cv_test"), 'wb') as filehandle:
		# 	pickle.dump(test_x_encodings_micro, filehandle)
		# with open(os.path.join(cv_dir, "best_rule_micro"), 'wb') as filehandle:
		# 	pickle.dump(best_rule_micro, filehandle)
		# with open(os.path.join(cv_dir, "test_encoding_cv_test"), 'rb') as filehandle:
		#     test_encoding_micro = pickle.load(filehandle)
		# with open(os.path.join(cv_dir, "best_rule_micro"), 'rb') as filehandle:
		#     best_rule_micro = pickle.load(filehandle)
		# with open(os.path.join(cv_dir, "arg_list_cv_test1.data"), 'rb') as filehandle:
		#     test_arg_list = pickle.load(filehandle)

		# decode_x(test_encoding_micro[best_rule_micro], test_arg_list)

		# with open(os.path.join(cv_dir, arg_list_fn), 'rb') as filehandle:
		#     arg_list = pickle.load(filehandle)	
		# with open(os.path.join(cv_dir, pos_list_fn), 'rb') as filehandle:
		#     pos_list = pickle.load(filehandle)		
		# print(arg_list[77 - train_x_indicies[0]])
		# print(pos_list[77 - train_x_indicies[0]])
	# test_encodings = apply_rules_test(pos_list[0:2], dep_list[0:2], arg_list[0:2], [], best_rule)
	# decode_x(test_encodings, arg_list)

	# var = json.loads('{"0": {"train": ["T124.EAGER.2.Night.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T124.EAGER.1.Ivan.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T130.EAGER.3.King.Final.xlsx"]}, "1": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T124.EAGER.2.Night.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx"]}, "2": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.2.Night.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T124.EAGER.3.TheName.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx"]}, "3": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.2.Night.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T125.EAGER.1.Sleepyhollow.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx"]}, "4": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.2.Night.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T125.EAGER.2.Ministersveil.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx"]}, "5": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.2.Night.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T125.EAGER.3.Poems.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx"]}, "6": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.2.Night.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T126.EAGER.1.Lordflies.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx"]}, "7": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.2.Night.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T126.EAGER.2.Mockingbird.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx"]}, "8": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.2.Night.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T127.EAGER.1.Heartdark.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"], "test": ["T126.EAGER.3.Mockingbird.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx"]}, "9": {"train": ["T124.EAGER.1.Ivan.Final.xlsx", "T124.EAGER.2.Night.Final.xlsx", "T124.EAGER.3.TheName.Final.xlsx", "T125.EAGER.1.Sleepyhollow.Final.xlsx", "T125.EAGER.2.Ministersveil.Final.xlsx", "T125.EAGER.3.Poems.Final.xlsx", "T126.EAGER.1.Lordflies.Final.xlsx", "T126.EAGER.2.Mockingbird.Final.xlsx", "T126.EAGER.3.Mockingbird.Final.xlsx", "T127.EAGER.2.Scarlet.Final.xlsx", "T127.EAGER.3.Shakespeare.Final.xlsx", "T128.EAGER.1.Mockingbird.Final.xlsx", "T128.EAGER.2.Wicked.Final.xlsx", "T128.EAGER.3.Littleprince.Final.xlsx", "T129.EAGER.1.Immortallife.Final.xlsx", "T129.EAGER.2.Crucible.Final.xlsx", "T129.EAGER.3.Wild.Final.xlsx", "T130.EAGER.1.OfMice.Final.xlsx", "T130.EAGER.3.King.Final.xlsx", "T131.EAGER.2.YellowWall.Final.xlsx", "T131.EAGER.3.Antigone.Final.xlsx", "T132.EAGER.1.Witches.Final.xlsx", "T132.EAGER.2.Crucible.Final.xlsx", "T132.EAGER.3.Wallflower.Final.xlsx", "T133.EAGER.1.Bleachers.Final.xlsx", "T133.EAGER.2.JFK.Final.xlsx"], "test": ["T127.EAGER.1.Heartdark.Final.xlsx", "T130.EAGER.2.Fahrenheit.Final.xlsx", "T133.EAGER.3.Mockingbird.Final.xlsx"]}}')
	# print(json.dumps(var, indent=2))






