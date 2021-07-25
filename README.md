# Turn Segmentation LSTM Documentation
### Author: Calvin Yu
### Dates: September 2020 - August 2021
## Description
This document contain a description of the main parts of the code used for turn segmentation from preprocessing the turns to evaluating the predictions from the model. Descriptions of how to run the code along with the necessary tools and environments are also provided.

## GPU Access
### Magnolia Server


## Environment
### Anaconda Environment (Miniconda3)
* Version: 4.9.2
* Instructions:
  * Install Miniconda3 
  ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```
  * Run the installed script and restart the shell
  ```bash Miniconda3-latest-Linux-x86_64.sh```
  * Create a tmux session (first tmux session)
  ```tmux new-session -s session_name```
  * Cd in to access the ***environment.yml*** file
  *  Access the  ***TurnSegmentation*** environment
  ```conda env create --file environment.yml```

## Other requirements
### CoreNLPParser
* Description: the tokenization class from ***nltk*** used to tokenize sentences, a server is required to run in the background since the tokenized turns have not been saved
* Instructions: 
  * Exit the current tmux session and create another tmux session (second tmux session)
  * Retrieve CoreNLP zip from server
  ```wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip```
  * Cd into the ***stanford-corenlp-full-2018-02-27*** folder
  * Run the server in the background 
  ```java -mx3g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer```

## Main Code Files
Pitt Box Directory: ***Turn_Segmentation_Calvin_Yu/LSTM_Approach/LSTM_Version_2***
* lstm\_pipeline\_python_v2.py
  * Contains the embedding model, neural network model, the dataset class, training, testing, and is the driver code
	  * Embedding models:
		  * Because of the all the different variants that can be employed, the embedding model is chosen by uncommenting/commenting the lines # - #
		  * Gensim 
			  * Embedding Size: 300
			  * Pre-trained Wiki News Fast-text (Default)
			  * Pre-trained GoogleNews Word2Vec (Baseline)
			  * Pre-trained Wiki Glove
		  * HuggingFace Transformers
			  * Embedding Size: 768
			  * DistilBert
	* Neural network model (Bi-LSTM)
		* Hyperparameters determined through cross validation in lstm_pipeline_python.py
		* Hidden size: 100
		* Number of Layers: 1
		* Output size/Number of classes: 3
	* Training
		* Hyperparameters determined through cross validation in lstm_pipeline_python.py
		* Epochs: 15
		* Learning Rate: 3e-3
		* Optimizer: Adam
		* Loss: Cross Entropy

* pipeline_v2.py
  * Contains various helper functions used in preprocessing and evaluation, check comments for purpose
  * Data exploration
	  * get_gold_stats()
	  * generate_trigrams()
	  * generate_bigrams()
  * Data pre-process
	  * preprocess_chunk()
	  * compress_parens()
	  * encode_y()
	  * make_arg_list()
  * Evaluation
	  * check_for_counts_thresholding()
	  * verify()
	  * segment_counter_1() - segment_counter_4()
	  * gold_counter_x()
	  * gold_counter_y()
	  * print_segments()
	  * decode_x()
	  * class_report_segment_level()
	  * class_report_token_level()
	  * conf_mat_segment_level()
	  * conf_mat_token_level()
  * Post-processing predictions
	  * replace_lower_postprocess()
	  * replace_upper_postprocess()
	  * bigram_postprocess()
  * O-insertion
	  * check_space()
	  * check_copy()
	  * check_punc()
	  * insert_Os()
  * Other helper functions
	  * num_to_tag()
	  * tag_to_num()
	  * write_list()

## Experiments and Results
Pitt Box Directory: ***Turn_Segmentation_Calvin_Yu/LSTM_Approach/LSTM_Version_2/Analysis and Results/Experiments***
### Main Experiments
  * Baseline Model
	  * Evaluated the LSTM using a pre-trained GoogleNews Word2Vec Gensim model, token-level and segment-level confusion matrices and classification reports
  * Baseline vs. Removing Parens from Baseline
	  * Compared the baseline model against the baseline model without parens in the turns at talk, token-level and segment-level confusion matrices and classification reports
  * Changing Gensim Word Embeddings Models
	  * Compared various Gensim models and trained one on a corpus, token-level and segment-level confusion matrices and classification reports
  * TARGER vs Fast-text Segment Level Comparison
	  * Compared TARGER https://github.com/uhh-lt/targer classification on the segment level with the top performer from the embedding models experiment

## Running the code
My current of keeping track of how I have been running the code is by writing the command in a shell script, see ***proc.sh*** as an example. The following python commands are run in the first tmux session.
### Argument Parsing
* Required Arguments:
  * t: the procedure to be done ("train" or "test")
  * p: the file path to the data
  * x: the name of the experiment

* Optional Flags:
  * the arguments depend on if training or testing is done or can be used during both 
  * m: flag used to save a statistics file containing confusion matrices and classification reports on the token-level and segment-level in a csv file
  * d: flag used to decode the predictions and save the decodings in a text file
  * e: flag used to save the edge case instances from O-insertion in a text file
  * w: flag used to save the weights of the model during training
 
### Example of training on a dataset
* Train the model on ***DT_2019*** and saves all the corresponding files
```python -W ignore lstm_pipeline_python_v2.py -t "train" -p "../Data/DT_2019" -x "fasttext_experiment"  -m -e -w``` 
* Notes:
	* the ***-w*** flag is required if testing follows training and the same weights are used
	* the ***-d*** flag is not used here and is reserved for testing

### Example of testing on a dataset
* Test the model on ***DT_2020*** and saves all the corresponding files 
```python -W ignore lstm_pipeline_python_v2.py -t "test" -p "../Data/DT_2020" -x "fasttext_experiment"  -m -d -e ``` 
* Notes:
	* the ***-w*** flag is never used here
	* the experiment folder is the same to access the saved weights file
