##########################################################################################
#                                  sentiment (archived)                                  #
########################################################################################## 


url <- "https://cran.r-project.org/src/contrib/Archive/sentiment/sentiment_0.2.tar.gz"
pkgFile <- "sentiment_0.2.tar.gz"
download.file(url = url, destfile = pkgFile)

# Install dependencies

install.packages(c("Rstem"))

# Install package
install.packages(pkgs=pkgFile, type="source", repos=NULL)


##########################################################################################
#                                  RTextTools trial                                      #
########################################################################################## 


# build the data to specify response variable, training set, testing set.

library(RTextTools)
library(tm)
library(SnowballC)  
library(dplyr)

doc_matrix <- create_matrix(Sample_data$Text, language="english", removeNumbers=TRUE,
                            stemWords=TRUE, removePunctuation = TRUE) #, removeSparseTerms=.998)

container <- create_container(doc_matrix, Sample_data$Sentiment, trainSize=1:450,
                              testSize=451:654, virgin=FALSE)

#container = create_container(matrix, as.numeric(as.factor(Sample_data[,2])),
#                             trainSize=1:450, testSize=451:654,virgin=FALSE)

models = train_models(container, algorithms=c("MAXENT" , "SVM", "RF", "BAGGING", "TREE"))

results = classify_models(container, models)

# accuracy table
table(as.numeric(as.factor(Sample_data[451:654, 2])), results[,"FORESTS_LABEL"])
table(as.numeric(as.factor(Sample_data[451:654, 2])), results[,"MAXENTROPY_LABEL"])

# recall accuracy
recall_accuracy(as.numeric(as.factor(Sample_data[451:654, 2])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.factor(Sample_data[451:654, 2])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric(as.factor(Sample_data[451:654, 2])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric(as.factor(Sample_data[451:654, 2])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric(as.factor(Sample_data[451:654, 2])), results[,"SVM_LABEL"])

# model summary
analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)
analytics@ensemble_summary

N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")