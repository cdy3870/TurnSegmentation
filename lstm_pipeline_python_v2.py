##### LIBRARIES #####
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader
from pipeline_v2 import *
from gensim.models import Word2Vec, KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import os
import glob
import matplotlib.pyplot as plt
import time
import copy
import collections
import argparse
import chardet
import gensim.downloader as api
from gensim.models.wrappers import FastText
from transformers import DistilBertTokenizer, DistilBertForTokenClassification, DistilBertModel, BertTokenizer, BertModel

def check_insertions(tokens_modified, encodings_y):
	for i in range(len(encodings_y)):
		if encodings_y[i] == 1:
			tokens_modified.insert(i, "<outside>")

	return tokens_modified


def modify_encoding(encoding, tokens_db, tokens_modified):
	"""
	For transformers usage, modifies an encoding and tokenized turn
	given the transformer tokenization, includes [CLS] and [SEP] insertions, 
	and split token insertions

	encoding: the encoding to be modified
	tokens_db: db = distilbert, but any tokenized sentence can be provided
	tokens_modified: the tokenized turn to be modified

	returns: None
	"""

	##### CLS AND SEP TOKENS ######
	cls_index = 0
	sep_index = len(encoding) + 1
	# Insert extra tokens/encodings
	tokens_modified.insert(cls_index, "[CLS]")
	tokens_modified.insert(sep_index, "[SEP]")
	encoding.insert(cls_index, 1)
	encoding.insert(sep_index, 1)

	# ##### SPLIT TOKENS #####
	# split_tokens = []
	# first_split = True
	# count = 0
	# for i in range(len(tokens_db)):
	# 	if tokens_db[i][0] == "#" and tokens_db[i][1] == "#":
	# 		if tokens_db[i - 1][0] != "#":
	# 			split_tokens.append(tokens_db[i - 1])
	# 			count += 1
	# 		split_tokens.append(tokens_db[i])

	# 		# Insert extra tokens/encodings
	# 		tokens_modified.insert(i, "[SPL]")
	# 		encoding.insert(i, 2)

	# 	elif tokens_db[i][0] == "'":
	# 		# Insert extra tokens/encodings
	# 		tokens_modified.insert(i + 1, "[SPL]")
	# 		encoding.insert(i, 2)


	# split_tokens.append(count)
	# return split_tokens

def get_max_length(tokens_x, encodings_y):
	"""
	Gets the max length of the tokenized turns and encodings provided

	tokens_x: the tokenized turns
	encodings_y: the encoded turns
	
	returns: the max length
	"""

	max_len = 0
	for i in range(len(tokens_x)):
		if len(tokens_x[i]) > max_len:
			max_len = len(tokens_x[i])
		if len(encodings_y[i]) > max_len:
			max_len = len(encodings_y[i])
	
	return max_len			

def pad(tokens_x, encodings_y, max_len):
	"""
	Pads the tokenized turn and encoding to the max length

	tokens_x: the tokenized turns
	encodings_y: the encoded turns
	max_len: the max length
	
	returns: None
	"""

	for i in range(len(encodings_y)):
		if len(encodings_y[i]) <= max_len:
			missing_len_arg = max_len - len(tokens_x[i])
			missing_len_encode = max_len - len(encodings_y[i])

			# Zero padding at the end
			encodings_y[i].extend([1] * (missing_len_encode))
			tokens_x[i].extend(["[PAD]"] * (missing_len_arg))
		else:
			# remove beginning and end if exceeds max length ([CLS] and [SEP] from transformer tokenization):
			for j in range(len(encodings_y[i]) - max_len):
				del encodings_y[i][len(encodings_y[i]) - 1]
				del tokens_x[i][len(encodings_y[i]) - 1]

def create_embeddings_x_hf(raw_args, max_len, tokens_modified, encodings_y):
	"""
	Creates embeddings using the distilbert transformer
	
	raw args: a list of raw, untokenized sentences
	max_len: the max length
	encodings_y: the encoded turns
	tokens_modified: the list of tokenized turns modified using this func.
	
	returns: a torch tensor containing the embeddings
	"""


	# print("testing with all arguments")
	tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased')
	model = BertModel.from_pretrained('distilbert-base-cased', output_hidden_states = True)

	input_ids = []
	split_tokens_list = []

	for i in range(len(raw_args)):
		# Print raw sentence/turn at talk
		# print("Sentence: ", end="")
		# print(raw_args[i])

		# Get input ids from transformer tokenization
		encode_dict = tokenizer.encode_plus(raw_args[i], truncation=True, max_length=max_len, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')
		input_ids.append(encode_dict['input_ids'])

		# Print the actual tokens given those input ids
		tokens_db = tokenizer.convert_ids_to_tokens(encode_dict['input_ids'][0].tolist())
		# print("DistBert Tokenization: ", end="")
		# print(tokens_db)

		# Modify current tokenization and encodings based on transformer tokens
		split_tokens = modify_encoding(encodings_y[i], tokens_db, tokens_modified[i])

		# Add padding to tokens and encodings after tokenization
		pad([tokens_modified[i]], [encodings_y[i]], max_len)
		# print(split_tokens)
		split_tokens_list.append(split_tokens)

		# Show similarity between the two
		# print("Modified Tokenization: ", end="")
		# print(tokens_modified[i])	



		# print("\n")
		

	# Feed input ids through DistilBert base model
	input_ids = torch.cat(input_ids, dim=0)
	model.eval()
	with torch.no_grad():
		outputs = model(input_ids)

	# print(len(outputs["hidden_states"]))
	# print(outputs["hidden_states"][0].size())

	num_layers = 7
	last_four = 4
	embedding_size = 768

	# x = outputs["hidden_states"][num_layers - 2]

	x = torch.zeros(len(raw_args), max_len, embedding_size)
	for i in range(num_layers - last_four, num_layers, 1):
		x = x.add(outputs["hidden_states"][i])
		# print(outputs["hidden_states"][i][0][0][:5])
		# print(x[0][0][:5])
		# print("\n")

	# print(x.size())
	# print(outputs["hidden_states"][0][0])
	# print(outputs["hidden_states"][1][0])
	# # print(outputs["hidden_states"][i][0][0][2])
	# print(outputs["hidden_states"][0][0].add(outputs["hidden_states"][1][0]))

	return x

def create_embeddings_x_gensim(tokens_x, max_len, model_path, model_type, embedding_size=300):
	"""
	Creates embeddings using a gensim embedding model

	tokens_x: the tokenized turns
	max_len: the max length
	model_path: the path to a gensim embedding model
	model_type: the type of model (fasttext, word2vec, glove)	
	embedding_size: the embedding size (default = 300)
	
	returns: a torch tensor containing the embeddings
	"""

	if model_type == 'w2v': 
		model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='replace')
	if model_type == 'other':
		model = api.load(model_path)

	x = torch.zeros(len(tokens_x), max_len, embedding_size)

	for i in range(len(tokens_x)):
		embeddings = torch.zeros(max_len, embedding_size)
		word_embedding_sum = np.zeros((1, embedding_size))
		for j in range(max_len):
			try:
				word_embedding = model.wv[tokens_x[i][j]]
				word_embedding_sum += word_embedding 
				embeddings[j] = torch.FloatTensor(word_embedding)
			except KeyError as e:
				if tokens_x[i][j] == "[PAD]":
					embeddings[j] = torch.zeros((1, embedding_size))
				else:
					embeddings[j] = torch.ones((1, embedding_size))

		# Using an averaged word embedding as replacement for unseen words
		num_unseen_words = 0
		for k in range(len(embeddings)):
			if torch.all(torch.eq(embeddings[k], torch.ones((1, 300)))):
				embeddings[k] = torch.from_numpy(word_embedding_sum/max_len)
				num_unseen_words+=1

		x[i] = embeddings

	return x

def create_embeddings_y(tokens_x, encodings_y, max_len):
	"""
	Creates encoding embeddings by converting tags to tensors

	tokens_x: the tokenized turns
	encodings_y: the encoded turns
	max_len: the max length
	
	returns: a torch tensor containing the encodings
	"""

	y = torch.zeros(len(tokens_x), max_len)
	for i in range(len(encodings_y)):
		y[i] = torch.FloatTensor(encodings_y[i])
	
	return y

def remove_pad(encodings_x, encodings_y, original_encodings, indices=""):
	"""
	Removes the padding from predictions for obtaining metrices, each 
	encoding is truncated to match the original encodings

	encodings_x: the predicted encodings
	encodings_y: the encoded turns (with padding)
	original_encodings: the original encodings (without padding)
	indices: the indices of data values (used in evaluation stage)
	
	returns: None
	"""

	increment = 0
	current_index = 0

	for i in range(len(encodings_y)):
		encodings_y[i] = encodings_y[i].numpy()
		encodings_x[i] = encodings_x[i].numpy()


		if indices == "":
			current_index = i
			original_encodings[i] = np.array(original_encodings[i])
			
		else:
			current_index = increment
			original_encodings[current_index] = np.array(original_encodings[current_index])

		if encodings_y[i].shape[0] != original_encodings[current_index].shape[0]:
			encodings_y[i] = encodings_y[i][:-(encodings_y[i].shape[0] - original_encodings[current_index].shape[0])]

			# Do the same with x encoding if it doesn't match y
			if encodings_x[i].shape[0] != encodings_y[i].shape[0]:
				encodings_x[i] = encodings_x[i][:-(encodings_x[i].shape[0] - encodings_y[i].shape[0])]

		encodings_x[i] = encodings_x[i].tolist()
		encodings_y[i] = encodings_y[i].tolist()

		increment += 1

class TranscriptDataset(Dataset):
	"""
	The dataset class used for training and testing pytorch models

	file_dir: a directory of files containing transcripts
	file_name: a single transcript file
	cv_dict: a dictionary of cross validation sets
	cv_value: the k value of cross validation
	cv_type: train or test for cross validation
	opened_file: the file where O-insertion edge cases are written to
	model_path: the file path of the gensim word embedding model
	model_type: the gensim embedding model type
	write_to_file: the flag to write to the edge case file or not
	embedding_type: huggingface (hf) or gensim
	
	returns: None
	"""
	def __init__(self, file_dir="", file_name="", cv_dict="", cv_value="", cv_type="", opened_file="", model_path="", model_type="", only_multi=False, write_to_file=False, embedding_type="gensim"):
		# If file path given in for cross validation json
		if cv_dict != "":
			file_dir = "/afs/cs.pitt.edu/usr0/cay44/private/Data/DT_2019"
			if cv_type == "train":
				files = [os.path.join(file_dir, filename) for filename in cv_dict[str(cv_value)]["train"]]
			else:
				files = [os.path.join(file_dir, filename) for filename in cv_dict[str(cv_value)]["test"]]
		# If file path given is a directory of csv
		elif file_dir != "":
			files = [os.path.join(file_dir, filename) for filename in os.listdir(file_dir) if filename.endswith(".xlsx")]  
			# files = [files[0]]
			# files = files[:-1]
			# print(len(files))

		# If file path is single file
		else:
			files = [file_name]

		tokens_x_list = []
		encodings_y_list = []
		segments_list = []
		raw_args_list = []
		ne_list = []

		# files = [files[0]]
		# Concatenation of transcripts
		for x in range(len(files)):
			out = convert_transcript(files[x])
			encodings_y, remove_list, _, segments = encode_y(to_TurnSegmentation(convert_transcript(files[x])))
			arg_list, tokens_x, _ = make_arg_list(to_TurnSegmentation(convert_transcript(files[x])))
			# insert_Os(tokens_x, segments, encodings_y, files[x], opened_file, out, write_to_file=write_to_file)
			encodings_y_list += encodings_y
			tokens_x_list += tokens_x
			segments_list += segments
			raw_args_list += arg_list 

		
		self.encodings = copy.deepcopy(encodings_y_list)

		# print("All lengths match: " + str(check_lengths(tokens_x_list, segments_list, encodings_y_list)))
		max_len = get_max_length(tokens_x_list, encodings_y_list)
		# print("Max Length: " + str(max_len))

		if embedding_type == "gensim":
			pad(tokens_x_list, encodings_y_list, max_len)
			self.x = create_embeddings_x_gensim(tokens_x_list, max_len, model_path, model_type)
		elif embedding_type == "hf":
			tokens_modified_list = copy.deepcopy(tokens_x_list)
			self.x = create_embeddings_x_hf(raw_args_list, max_len, tokens_modified_list, encodings_y_list)

		self.y = create_embeddings_y(tokens_x_list, encodings_y_list, max_len)
		self.arguments = raw_args_list
		

	def __len__(self):
		return self.x.size()[0]

	def __getitem__(self, index):
		indexed_x = self.x[index]
		indexed_y = self.y[index]

		return indexed_x, indexed_y


class BiLSTM(nn.Module):
	"""
	The Bi-LSTM class used for token classification

	input_size: the size of the embedding
	hidden_size: the number of units in the hidden layer
	num_classes: the number of tokens to be classified (IOB)
	num_layers: the number of layers in the model
	
	returns: None
	"""

	def __init__(self, input_size, hidden_size, num_classes, num_layers):
		super(BiLSTM, self).__init__()
		self.bi_lstm = nn.LSTM(input_size=input_size,
							hidden_size=hidden_size,
							num_layers=num_layers, 
							batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)

		
	def forward(self, x):
		out, final = self.bi_lstm(x)
		fc_out = self.fc(out)
		out = fc_out.view(-1, num_classes)
		return out

class BaseNN(nn.Module):

	def __init__(self, input_size, ):
		super(BaseNN, self).__init__()
		self.linear_and_relu = nn.Sequential(nn.Linear(input_size, 512),
											 nn.ReLU(),
											 nn.Linear(512, num_classes),
											 nn.ReLU())

	def forward(self, x):
		out = self.linear_and_relu(x)
		out = out.view(-1, num_classes)
		return out


def apply_word2vec_model(w2v_txt_file):
	"""
	Creates a gensim word2vec model given a file containing sentences

	w2v_txt_file: the file name of the corpus

	returns: the gensim model
	"""

	count = 0
	train_documents = []
	embedding_size = 300

	#read in opinion file for word2vec training
	with open(w2v_txt_file, encoding="utf8", errors='ignore') as fp:
		line = fp.readline()

		while line:
			# print(line)
			if line[12] == " ":
				line = line[12:]
			else:
				line = line[11:]	
			train_documents.append(preprocess_chunk(line))
			# print(train_documents[count])
			count += 1
			if count == 1500:
				break

			line = fp.readline()

	#train word2vec model
	model = Word2Vec(train_documents, size=embedding_size)
	w2v_model = "opinrank_word2vec_model.txt"

	#save model
	model.wv.save_word2vec_format(w2v_model, binary=True)

	return w2v_model


def train(params, model, train_loader, val_loader, val_indices, original_encodings, experiment_name, weights_file_name, save_weights_file=True, stats_path=""):
	"""
	Trains a token classification model with a validation set

	params: the hyperparameters of the model determined from cross validation
	model: the pytorch model used
	train_loader: the pytorch dataloader (train set)
	val_loader: the pytorch dataloader (validation set)
	val_indices: indices of the validation set (for padding removal purposes, since random_split randomizes selection)
	original_encodings: the original encodings without padding
	experiment_name: the name of the experiment
	weights_file_name: the name of the weights file saved
	save_weights_file: flag used to determine whether file is saved
	stats_path: the file path/name of saved metrics

	returns: None
	"""

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	total_loss = 0
	optimizer = torch.optim.Adam(model.parameters(),
								 lr=params["lr"])
	# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	progress_bar = tqdm(range(params["num_epochs"]))

	total_loss = 0
	best_f_score = 0
	num_correct = 0
	true_size = 0
	f_score_sum = 0
	f_score_count = 0

	encodings_y = []
	encodings_x = []

	for epoch in progress_bar:

		# print("Epoch: " + str(epoch))
		iteration = 0
		for phase in ('train', 'eval'):
			if phase == 'train':
				model.train()
				data_loader = train_loader
			else:
				model.eval()
				data_loader = val_loader

			for (_, data) in enumerate(data_loader):
				# if phase == 'eval':
				# 	print(count)		
				# 	count += 1
				optimizer.zero_grad()
				embeddings = data[0].to(device)
				iob_labels = data[1]
				iob_labels = iob_labels.view(-1).to(device)

				with torch.set_grad_enabled(phase=='train'):
					iob_labels_out = model(embeddings)

					loss = criterion(iob_labels_out, iob_labels.long())
					# print("Iter " + str(iteration) + " loss: " + str(loss.item()))
					iteration += 1
					if phase == 'train':
						loss.backward()
						optimizer.step()

				out, max_indices = torch.max(iob_labels_out, dim=1)
				num_correct += torch.sum(max_indices == iob_labels.long()).item()
				true_size += iob_labels.size()[0]



				
				if phase == 'train':
					total_loss += loss.item()

				if phase == 'eval' and epoch == params["num_epochs"] - 1:
					encodings_y.append(iob_labels.cpu())
					encodings_x.append(max_indices.cpu())

	# if phase == 'train':
	# 	train_accuracy = str(num_correct/true_size)
	# 	print("Train Accuracy: {}".format(train_accuracy)) 

	if phase == 'eval':
		val_accuracy = str(num_correct/true_size)
		print("Val Accuracy: {}".format(val_accuracy)) 

		if stats_path != "":
			# Gold counts for gold segmentation
			counts_y = [collections.Counter(original_encodings[x])[0] for x in val_indices]

			remove_pad(encodings_x, encodings_y, original_encodings, indices=val_indices)

			# Unbounded counting for prediction
			counts_x = segment_counter_2(encodings_x)

			combined_y = []
			combined_x = []
			list(map(combined_y.extend, encodings_y))
			list(map(combined_x.extend, encodings_x))	

			print(encodings_x[0])	

			##### TOKEN-LEVEL SAVING PRECISION AND RECALL TO CSV #####

			class_report_token_level(encodings_x, encodings_y, stats_path)

			##### TOKEN-LEVEL SAVING CONF MATRIX TO CSV #####

			token_mat = conf_mat_token_level(encodings_x, encodings_y, stats_path)

			# ##### SEGMENT-LEVEL SAVING PRECISION AND RECALL TO CSV ######
			class_report_segment_level(counts_x, counts_y, stats_path)

			# # ##### SEGMENT-LEVEL SAVING CONF MATRIX TO CSV ######

			segment_mat = conf_mat_segment_level(counts_x, counts_y, stats_path)

	if save_weights_file:
		best_model_weights = copy.deepcopy(model.state_dict())
		# print("saving weights file")
		torch.save({'model' : best_model_weights}, os.path.join(experiment_name, weights_file_name))


##### TRAIN ON FULL DATASET #####
def train_full(params, model, train_loader, experiment_name, weights_file_name):
	"""
	Same as train() but without validation phase

	params: the hyperparameters of the model determined from cross validation
	model: the pytorch model used
	train_loader: the pytorch dataloader (train set)
	experiment_name: the name of the experiment
	weights_file_name: the name of the weights file saved

	returns: None
	"""

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = BiLSTM(input_size=input_size,
				   hidden_size=bi_hidden_size,
				   num_classes=num_classes,
				   num_layers=params["num_layers"])
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	total_loss = 0
	optimizer = torch.optim.Adam(model.parameters(),
								 lr=params["lr"])
	# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	progress_bar = tqdm(range(params["num_epochs"]))

	total_loss = 0
	best_f_score = 0
	num_correct = 0
	true_size = 0
	f_score_sum = 0
	f_score_count = 0
	phase = "train"

	for epoch in progress_bar:
		iteration = 0

		for phase in ('train', 'eval'):
			if phase == 'train':
				model.train()
				data_loader = train_loader

		for (_, data) in enumerate(data_loader):
#             iob_labels = iob_labels.to(device)
			optimizer.zero_grad()
			embeddings = data[0].to(device)
			iob_labels = data[1]
			iob_labels = iob_labels.view(-1).to(device)

			with torch.set_grad_enabled(phase=='train'):
				iob_labels_out = model(embeddings)
				loss = criterion(iob_labels_out, iob_labels.long())

				iteration += 1
				if phase == 'train':
					loss.backward()
					optimizer.step()

			out, max_indices = torch.max(iob_labels_out, dim=1)
			num_correct += torch.sum(max_indices == iob_labels.long()).item()
			true_size += iob_labels.size()[0]
			
			if phase == 'train':
				total_loss += loss.item()

	# best_model_weights = copy.deepcopy(model.to(device).state_dict())
	best_model_weights = copy.deepcopy(model.state_dict())
	torch.save({'model' : best_model_weights}, os.path.join(experiment_name, weights_file_name))

def test(params, model, test_loader, experiment_name, weights_file_name):
	"""
	Tests a token classification model, obtains predicted encodings

	params: the hyperparameters of the model determined from cross validation
	model: the pytorch model used
	test_loader: the pytorch dataloader (test set)
	experiment_name: the name of the experiment
	weights_file_name: the name of the weights file saved

	returns: None
	"""

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	result_list = []
	word_list = []
	true_list = []
	num_correct = 0
	true_size = 0
	state_dict = torch.load(os.path.join(experiment_name, weights_file_name))
	model.load_state_dict(state_dict['model'])
	model.eval()

	for (_, data) in enumerate(test_loader):
		embeddings = data[0]
		iob_labels = data[1]
		iob_labels = iob_labels.view(-1)
		iob_labels_out = model(embeddings)
		out, max_indices = torch.max(iob_labels_out, dim=1)

		result_list.append(iob_labels)
		word_list.append(max_indices)

	return word_list, result_list


def main():
	parser = argparse.ArgumentParser(description="")

	parser.add_argument("-a", "--msg", type=str, help="description of task, for logging purposes")
	
	# required arguments
	parser.add_argument("-p", "--path", type=str, help="data file path for train or test", required=True)
	parser.add_argument("-t", '--train_or_test', type=str, help="training or testing (\"train\", \"test\")", required=True)
	parser.add_argument("-x", "--experiment_folder_name", type=str, help="name of experiment", required=True)

	# optional files to save
	parser.add_argument("-m", '--save_stats_file', action="store_true", help="computes standard metrics and conf matrices and saves file")
	parser.add_argument("-d", "--save_decoded_predictions", action="store_true", help="decodes the predictions and saves to a file")
	parser.add_argument("-e", '--save_edge_case_file', action="store_true", help="saves edge cases for O insertions")
	parser.add_argument("-w", '--save_weights_file', action="store_true", help="saves a weights file during training")

	args = parser.parse_args()

	print("Train or Test: " + str(args.train_or_test) + ", Data: " + str(args.path) + ", Experiment: " + str(args.experiment_folder_name))
	print("Task: " + str(args.msg))

	if args.save_stats_file:
		print("Stats file will be saved.")
	if args.save_decoded_predictions:
		print("Decoded predictions file will be saved.")
	if args.save_edge_case_file:
		print("Edge case file will be saved.")
	if args.save_weights_file:
		print("Weights file will be saved.")

	os.environ['CUDA_VISIBLE_DEVICES'] = '6'

	##### TRAINING AND TESTING PREREQS ######
	# Embedding type
	# embedding_type = "hf"
	embedding_type = "gensim"

	# Word2Vec/Fast-text/Glove Models
	em_model_type = "other"
	# em_model_type = "w2v"

	# em_model = apply_word2vec_model("opinrank.txt") 		  # Training w2v model
	# em_model = "GoogleNews-vectors-negative300.bin"		  # Pre-trained w2v model
	em_model = "fasttext-wiki-news-subwords-300"  		  # Pre-trained ft model
	# em_model = "glove-wiki-gigaword-300"					  # Pre-trained glove model	


	# Hyperparameters chosen based on previous cross-validation
	params = {"num_epochs": 15, "lr": 3e-3, "num_layers": 1}

	# Structure params
	num_classes = 3
	if embedding_type == "gensim":
		input_size = 300
	else:
		input_size = 768
	hidden_size = 100
	bi_hidden_size = hidden_size*2
	num_layers = 1

	# Model instantiation
	model = BiLSTM(input_size=input_size,
				   hidden_size=bi_hidden_size,
				   num_classes=num_classes,
				   num_layers=params["num_layers"])

	experiment_name = "../experiments/" + args.experiment_folder_name
	experiment_details_file_name = "experiment_details.txt"
	weights_file_name = 'best_model_weights.pt'
	stats_file_name = "stats_" + str(args.train_or_test) + ".csv"
	error_file_name = "edge_cases.txt"
	decode_file_name = "decoded_predictions.txt"

	if args.train_or_test == "train":
		if os.path.exists(experiment_name):
			files = glob.glob(experiment_name + "/*")
			for file in files:
				os.remove(file)
			os.rmdir(experiment_name)
		os.mkdir(experiment_name)

	if args.save_edge_case_file:
		opened_file = open(os.path.join(experiment_name, error_file_name), "w")
	if args.save_decoded_predictions:
		opened_file_decode = open(os.path.join(experiment_name, decode_file_name), "w")

	##### TRAINING ######
	if args.train_or_test == "train":
		opened_file_details = open(os.path.join(experiment_name, experiment_details_file_name), "w")
		opened_file_details.writelines(["Train Data: " + str(args.path) + "\n"])
		opened_file_details.writelines(["Embedding Model: " + str(em_model) + "\n"])
		opened_file_details.writelines(["Num Epochs: " + str(params["num_epochs"]) + "\n"])
		opened_file_details.writelines(["Learning Rate: " + str(params["lr"]) + "\n"])
		opened_file_details.writelines(["Num Layers: " + str(params["num_layers"]) + "\n"])
		opened_file_details.writelines(["Hidden Layer Size: " + str(hidden_size) + "\n"])

		file_dir = 	args.path
		get_gold_stats(file_dir)
		batch_size = 1
		train_percent = 0.6

		dataset = TranscriptDataset(file_dir=file_dir, opened_file=opened_file, model_path=em_model, model_type=em_model_type, write_to_file=args.save_edge_case_file, embedding_type=embedding_type)
		original_encodings = dataset.encodings
		train_size = int(train_percent*len(dataset))
		val_size = len(dataset) - train_size 
		train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_size, val_size))
		train_loader = DataLoader(train_dataset,
		                          batch_size=batch_size,
		                          shuffle=False) 
		val_loader = DataLoader(val_dataset,
		                          batch_size=batch_size,
		                          shuffle=False) 
		val_indices = val_dataset.indices

		if args.save_stats_file:
			with open(os.path.join(experiment_name, stats_file_name), 'w') as csvfile:
				csv_writer = csv.writer(csvfile) 
				csv_writer.writerow(["Stats"])
			train(params, model, train_loader, val_loader, val_indices, original_encodings, experiment_name, weights_file_name, save_weights_file=args.save_weights_file, stats_path=os.path.join(experiment_name, stats_file_name))
		else:
			train(params, model, train_loader, val_loader, val_indices, original_encodings, experiment_name, weights_file_name, save_weights_file=args.save_weights_file)

		opened_file_details.close()


	##### TESTING #####
	if args.train_or_test == "test":
		file_dir = args.path
		get_gold_stats(file_dir)
		full_x_encodings = []
		full_y_encodings = []
		full_arg_list = []
		final_output = []
		counts_y = []
		segment_list = []

		files = [os.path.join(file_dir, filename) for filename in os.listdir(file_dir) if filename.endswith(".xlsx")]
		file_names = [filename for filename in os.listdir(file_dir) if filename.endswith(".xlsx")]
		# files = [files[0]]
		for x in range(len(files)):
			stats_path = os.path.join(experiment_name, stats_file_name)

			# Gold Segment Counts for Y
			counts_y += gold_counter_y(to_TurnSegmentation(convert_transcript(files[x])), files[x])

			test_dataset = TranscriptDataset(file_name=files[x], opened_file=opened_file, model_path=em_model, model_type=em_model_type, write_to_file=args.save_edge_case_file, embedding_type="hf")
			test_loader = DataLoader(test_dataset,
			                          batch_size=1,
			                          shuffle=False)
			original_encodings = test_dataset.encodings

			test_x_encodings, test_y_encodings = test(params, model, test_loader, experiment_name, weights_file_name)
			test_arg_list = test_dataset.arguments	

			remove_pad(test_x_encodings, test_y_encodings, original_encodings)

			# check_for_counts_thresholding(test_x_encodings, gold_counter_x(test_x_encodings), test_arg_list, file_names[x])

			full_x_encodings = full_x_encodings + test_x_encodings
			full_y_encodings = full_y_encodings + test_y_encodings
			full_arg_list = full_arg_list + test_arg_list

			if args.save_decoded_predictions:
				decode_x(test_x_encodings, test_y_encodings, test_arg_list, segment_list, opened_file=opened_file_decode, write_to_file_decode=True, file_name=file_names[x], )

		# Gold Segment Counts for X
		bounded_counts_x, unbounded_counts_x = gold_counter_x(full_x_encodings)
		greater_than_3 = [unbounded_counts_x[i] for i in range(len(unbounded_counts_x)) if counts_y[i] == 3 and unbounded_counts_x[i] >= 3]
		print(collections.Counter(greater_than_3))
		print(len(greater_than_3))


		if args.save_stats_file:
			##### TOKEN-LEVEL SAVING PRECISION AND RECALL TO CSV #####

			class_report_token_level(full_x_encodings, full_y_encodings, stats_path, write_to_csv=True)

			##### TOKEN-LEVEL SAVING CONF MATRIX TO CSV #####

			token_mat = conf_mat_token_level(full_x_encodings, full_y_encodings, stats_path, write_to_csv=True)

			# ##### SEGMENT-LEVEL SAVING PRECISION AND RECALL TO CSV ######
			class_report_segment_level(bounded_counts_x, counts_y, stats_path, write_to_csv=True)

			# # ##### SEGMENT-LEVEL SAVING CONF MATRIX TO CSV ######

			segment_mat = conf_mat_segment_level(bounded_counts_x, counts_y, stats_path, write_to_csv=True)

			# verify(segment_mat, token_mat)

			# with open(stats_path, 'w') as csvfile:
			# 	csv_writer = csv.writer(csvfile) 
			# 	csv_writer.writerow(["Stats"])

			# lower_bound = 1
			# upper_bound = 3
			# for i in range(len(full_x_encodings)):
			# 	replace_lower_postprocess(full_x_encodings[i], unbounded_counts_x, i, lower_bound)
			# 	replace_upper_postprocess(full_x_encodings[i], unbounded_counts_x, i, upper_bound)
			# 	# bigram_postprocess(full_x_encodings[i])

			# # regenerate conf matrices
			# class_report_token_level(full_x_encodings, full_y_encodings, stats_path)
			# token_mat = conf_mat_token_level(full_x_encodings, full_y_encodings, stats_path)
			# class_report_segment_level(unbounded_counts_x, counts_y, stats_path)
			# segment_mat = conf_mat_segment_level(unbounded_counts_x, counts_y, stats_path)

			# if segment_mat.shape[0] > upper_bound:
			# 	verify(segment_mat, token_mat, include_zero=True)
			# else:
			# 	verify(segment_mat, token_mat, include_zero=False)

	# Close text files
	if args.save_edge_case_file:
		opened_file.close()
	if args.save_decoded_predictions:	
		opened_file_decode.close()


if __name__ == "__main__":
	print("\n")
	# parser = argparse.ArgumentParser(description="")
	# parser.add_argument("-a", "--msg", type=str, help="description of task", required=True)
	# args = parser.parse_args()
	# print("Task: " + str(args.msg))

	# Testing token modification
	# file_dir = "../Data/DT_2019"
	# files = [os.path.join(file_dir, filename) for filename in os.listdir(file_dir) if filename.endswith(".xlsx")]
	# files = [files[0]]
	# encodings_y_list = []
	# tokens_x_list = []
	# raw_arg_list = []
	# tokens_modified_list = []
	# for x in range(len(files)):
	# 	out = convert_transcript(files[x])
	# 	encodings_y, remove_list, _, segments = encode_y(to_TurnSegmentation(convert_transcript(files[x])))
	# 	arg_list, tokens_x, _ = make_arg_list(to_TurnSegmentation(convert_transcript(files[x])))
	# 	insert_Os(tokens_x, segments, encodings_y, files[x], None, out, write_to_file=False)
	# 	encodings_y_list = encodings_y_list + encodings_y
	# 	tokens_x_list = tokens_x_list + tokens_x
	# 	tokens_modified = copy.deepcopy(tokens_x)
	# 	tokens_modified_list += check_insertions(tokens_modified, encodings_y)
	# 	raw_arg_list = raw_arg_list + arg_list 

	# max_len = get_max_length(tokens_modified_list, encodings_y_list)
	# create_embeddings_x_hf(raw_arg_list, max_len, tokens_modified_list, encodings_y_list)
	# print("\n")


	main()
























	


# COMMENT START

	##### CROSS-VALIDATION #####
	# if args.cv_or_train_or_test:
	# 	cv_dict = json.loads()	
	# 	batch_size = 1

	# 	if args.save_stats_file:
	# 		with open(os.path.join(experiment_name, stats_file_name), 'w') as csvfile:
	# 			csv_writer = csv.writer(csvfile) 
	# 			csv_writer.writerow(["Stats"])

	# 	for i in range(len(cv_dict.keys()))
	# 		train_dataset = TranscriptDataset(cv_dict=cv_dict, cv_value=i, cv_type="train")
	# 		val_dataset = TranscriptDataset(cv_dict=cv_dict, cv_value=i, cv_type="test")
	# 		train_loader = DataLoader(train_dataset,
	# 		                          batch_size=batch_size,
	# 		                          shuffle=False)
	# 		val_loader = DataLoader(val_dataset,
	# 		                        batch_size=batch_size,
	# 		                        shuffle=False)

	# 		if args.save_stats_file:
	# 			train(params, train_loader, val_loader, val_indices, original_encodings, experiment_name, weights_file_name, stats_path=os.path.join(experiment_name, stats_file_name))


	# # print(key_words_x)
	# # print(key_words_y)
	# # print(collections.Counter(key_words_x))
	# # print(collections.Counter(key_words_y))


	# # # # print("x")
	# # # # print(counts_x)
	# # print("Gold Standard")
	# # print("Segment Count Method 1 (lower (1) and upper bound (4))")
	# # num_segments = segment_counter_1(full_y_encodings)
	# # print("Gold Matches Counter: " + str(counts_y == num_segments))
	# # num_segments = segment_counter_1(full_x_encodings)
	# # print("Pred Matches Counter: " + str(counts_x == num_segments))
	# # print("\n")

	# # print("Segment Count Method 2 (no lower and upper bound)")
	# # num_segments = segment_counter_2(full_y_encodings)
	# # print("Gold Matches Counter: " + str(counts_y == num_segments))
	# # num_segments = segment_counter_2(full_x_encodings)
	# # print("Pred Matches Counter: " + str(counts_x == num_segments))
	# # print("\n")


	# # print("Segment Count Method 3 (only lower bound (1))")
	# # num_segments = segment_counter_3(full_y_encodings)
	# # print("Gold Matches Counter: " + str(counts_y == num_segments))
	# # num_segments = segment_counter_3(full_x_encodings)
	# # print("Pred Matches Counter: " + str(counts_x == num_segments))
	# # print("\n")

	# # print("Segment Count Method 4 (only upper bound (4))")
	# # num_segments = segment_counter_4(full_y_encodings)
	# # print("Gold Matches Counter: " + str(counts_y == num_segments))
	# # num_segments = segment_counter_4(full_x_encodings)
	# # print("Pred Matches Counter: " + str(counts_x == num_segments))
	# # print("\n")


	# # # # print(full_y_encodings)
	# # # # print(full_arg_list)

	# # token_mat = np.array([[1553, 0, 232], [0, 0, 10], [449, 41, 106588]])
	# # segment_mat = np.array([[969, 109, 120, 0], [148,  26,  61,   0], [ 15,   8, 11,   0], [  1,   2,   1,   0]])

	# print(generate_bigrams(full_x_encodings))
	# print(generate_trigrams(full_x_encodings))


	# ##### SEGMENT COUNT VERIFICATION #####
	# # initial verification
	# verify(segment_mat, token_mat)

	# lower_bound = 1
	# upper_bound = 3
	# for i in range(len(full_x_encodings)):
	# 	replace_lower_postprocess(full_x_encodings[i], unbounded_counts_x, i, lower_bound)
	# 	replace_upper_postprocess(full_x_encodings[i], unbounded_counts_x, i, upper_bound)
	# 	bigram_postprocess(full_x_encodings[i])

	# # regenerate conf matrices
	# class_report_token_level(full_x_encodings, full_y_encodings, "multi_report.csv")
	# token_mat = conf_mat_token_level(full_x_encodings, full_y_encodings)
	# class_report_segment_level(unbounded_counts_x, counts_y, "multi_report.csv")
	# segment_mat = conf_mat_segment_level(unbounded_counts_x, counts_y, "multi_report.csv")

	# # re-verify
	# verify(segment_mat, token_mat)

# COMMENT END






