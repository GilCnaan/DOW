################################
###### READ ME           #######
###### By Chris Beegle   #######
###### Drexel University #######
################################

#/////////////
#/ CONTENTS //
#/////////////

This code was developed using Python 3. It also requires the following packages to be installed:
pandas, re, time, nltk, sklearn, numpy, textblob

This folder contains:
Three Python Jupyter notebooks:
	Homecare.ipynb
	Feature Building For LDA.ipynb
	LDA and NMF.ipynb

Three python scripts with the same code as the notebooks above:
	Homecare.py
	Feature Building For LDA.py
	LDA and NMF.py
	
Three datasets:
	wipes_reviews.csv
	Ratings_Only_Reviews.csv
	wipes_market_item_name_terms.csv
	

#/////////////////
#/ INSTRUCTIONS //
#/////////////////

1) Ensure all files are saved in the same directory and all packages and dependencies are intalled.
2) To run the fiels, paste the code into your IDE or open the notebooks in Jupyter.
3) Run 'Homecare' (.py or ipynb) first. This will output 3 .csv files:
				home_products.csv    <- The data in this file is used by the subsequent script in step 4
				count_of_products.csv
				products_with_more_than_ten_reviews.csv
4) Run 'Feature Building For LDA' (.py or ipynb). This uses the output from 'Homecare.py' (named 'home_products.csv')
				This script is creating new columns in the dataframe that are variations on the original review text.
				These new feature columns can be used for analysis with LDA and other algorithms to determine if better topics/attributes can be identified.
5) The last file to run is 'LDA and NMF' (.py or ipynb).
				This code contains two models and is set up to automatically read in the data and train an LDA and NMF (Non-Negative Matrix Factorization) model. 
				There is a section for each model at the end of the script with many parameters that can be tuned to affect the output of the topics. 
				Comments are included with each parameter (taken from the SKLearn website). This code is designed to be run multiple times, iteratively adjusting the parameters
				and manually inspecting the topics and attributes to assess for furthur use.
				
				*******PARAMETER TUNING**********
				When running this script, each model section has 12 lines of code that set the data_sample (for lda or nmf) to a specific feature set created in the previous
				script. To change which feature set the model uses, simply uncomment (remove the #) from one of the lines. After the data sample section you will find a group
				of variables (such as lda_features) that are set to an integer or float value. By experimenting with these values, you will change the output of the model and the
				topics it creates. 