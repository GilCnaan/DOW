############################
# This time, with feeling! #
############################

#load text mining library
library(tm)


#set working directory (modify path as needed)
setwd("E:/Classes/BUSN710/DOW Project/Data") #"C:/Users/BravoOne/Documents/Graduate School/Classes/BUSN710/DOW Project/Data"


#Import Data
p_napp <- readLines("E:/Classes/BUSN710/DOW Project/Data/Filtered Data.txt") # C:/Users/BravoOne/Documents/Graduate School/Classes/BUSN710/DOW Project/Data/Filtered Data.txt


p_napp <- paste(p_napp, collapse = " ")
p_napp <- strsplit(p_napp, "&&&")[[1]] # create substrings in your text if it's comma delimited
length(p_napp)

#inspect a particular document in corpus
writeLines(as.character(p_napp[[30]]))

#create corpus from vector
docs <- Corpus(VectorSource(p_napp))

#start preprocessing
#Transform to lower case
docs <-tm_map(docs,content_transformer(tolower))


#remove potentially problematic symbols
toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, " ", x))})
docs <- tm_map(docs, toSpace, "-")
docs <- tm_map(docs, toSpace, "'")
docs <- tm_map(docs, toSpace, "'")
docs <- tm_map(docs, toSpace, ".")
docs <- tm_map(docs, toSpace, """)
docs <- tm_map(docs, toSpace, """)


#remove punctuation
docs <- tm_map(docs, removePunctuation)
#Strip digits
docs <- tm_map(docs, removeNumbers)
#remove stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
#remove whitespace
docs <- tm_map(docs, stripWhitespace)
#Good practice to check every now and then
writeLines(as.character(docs[[30]]))
#Stem document
docs <- tm_map(docs,stemDocument)


#fix for when changing differences in "english" origin (eg, Australian English)
#docs <- tm_map(docs, content_transformer(gsub),
#               pattern = "organiz", replacement = "organ")

#define and eliminate all custom stopwords
#this was done for all words with a frequency > 200 (ref: word_freq output)
myStopwords <- c( "and" , "the"
#                 "wipe" , "use" , "get" , "just" , "dont" , "brand" , "make" , "easi" , "keep" , "tri" ,
#                 "will" , "also" , "buy" , "much" , "need" , "alway" , "job" , "even" ,
#                 "come" , "ive" , "now" , "purchas" , "review" , "without" , "free" ,
#                 "around" , "know" , "way" , "doesnt" , "done" , "theyr" , "want" , 
#                 "find" , "receiv" , "lot" , "say" , "definit" , "got" , "think" , "expeci" ,
#                 "ever" , "give" , "differ" , "right" , "cant" , "amazon" , "thank" , "warmer" ,
#                 "etc" , "came" 
                )
docs <- tm_map(docs, removeWords, myStopwords)
#inspect a document as a check
writeLines(as.character(docs[[30]]))


#Create document-term matrix
dtm <- DocumentTermMatrix(docs)
#convert rownames to filenames
rownames(dtm) <- filenames
#collapse matrix by summing over columns
freq <- colSums(as.matrix(dtm))
#length should be total number of terms
length(freq)
#create sort order (descending)
ord <- order(freq,decreasing=TRUE)
#List all terms in decreasing order of freq and write to disk
freq[ord]
write.csv(freq[ord],"word_freq_2.csv")




#load topic models library
library(topicmodels)


#Remove blank rows from the DTM
rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
dtm.new   <- dtm[rowTotals> 0, ]           #remove all docs without words

#Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE


#Number of topics
k <- 50

#That done, we can now do the actual work - run the topic modelling algorithm on our corpus. Here is the code:
#Run LDA using Gibbs sampling
ldaOut <-LDA(dtm.new,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))



#set working directory for results, or just # out if you want to keep the same as above
setwd("G:/Classes/BUSN710/DOW Project/Data/Results") #"C:/Users/BravoOne/Documents/Graduate School/Classes/BUSN710/DOW Project/Data"

#write out results
#docs to topics
ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics_2.csv"))


#top 6 terms in each topic
ldaOut.terms <- as.matrix(terms(ldaOut,6))
write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms_2.csv"))


#probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))


#Find relative importance of top 2 topics
topic1ToTopic2 <- lapply(1:nrow(dtm.new),function(x) sort(topicProbabilities[x,])[k]/sort(topicProbabilities[x,])[k-1])


#Find relative importance of second and third most important topics
topic2ToTopic3 <- lapply(1:nrow(dtm.new),function(x)
  sort(topicProbabilities[x,])[k-1]/sort(topicProbabilities[x,])[k-2])


#write to file
write.csv(topic1ToTopic2,file=paste("LDAGibbs",k,"Topic1ToTopic2.csv"))
write.csv(topic2ToTopic3,file=paste("LDAGibbs",k,"Topic2ToTopic3.csv"))


