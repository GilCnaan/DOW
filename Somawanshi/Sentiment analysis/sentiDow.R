library(RTextTools)
library(e1071)
library(plyr)
library(stringr)

Sample <- read.csv("E:\\Drexel\\Course-Study\\Fall 2016 Sep19-Dec10\\BUSN-710\\Dows-Project\\Rated review\\Sample Set Full.csv")

wipes_reviews <- read.csv("E:\\Drexel\\Course-Study\\Fall 2016 Sep19-Dec10\\BUSN-710\\Dows-Project\\wipes_reviews.csv")

Results<-data.frame(X1=NA,vNegTerms_Count=NA,negTerms_Count=NA,posTerms_Count=NA,vPosTerms_Count=NA)

sentimentScore <- function(sentences,vNegTerms, negTerms, posTerms, vPosTerms)
{
  
  #final_scores <- matrix('', 0, 5)
  
  end_score<-data.frame(X1=NA,vNegTerms_Count=NA,negTerms_Count=NA,posTerms_Count=NA,vPosTerms_Count=NA)
  i=1;scores<-0
  for(i in 1:1)
  {
    final_scores<-data.frame()
    
    #final_scores<-data.frame(X1=NA,vNegTerms_Count=NA,negTerms_Count=NA,posTerms_Count=NA,vPosTerms_Count=NA)
    scores <-as.data.frame(lapply(sentences[i], function(sentence, vNegTerms, negTerms, posTerms, vPosTerms){
      
      initial_sentence <- sentence
      initial_sentence<-gsub("*\\n", '', sentence)
      #initial_sentence<-gsub("newrow", '', sentence)
      
      #remove unnecessary characters and split up by word 
      sentence <- gsub('[[:punct:]]', '', sentence)
      sentence <- gsub('[[:cntrl:]]', '', sentence)
      sentence <- gsub('\\d+', '', sentence)
      sentence <- gsub("*\\n", '', sentence)
      sentence <- tolower(sentence)
      wordList <- str_split(sentence, '\\s+')
      words <- unlist(wordList)
      #build vector with matches between sentence and each category
      vPosMatches <- match(words, vPosTerms)
      posMatches <- match(words, posTerms)
      vNegMatches <- match(words, vNegTerms)
      negMatches <- match(words, negTerms)
      #sum up number of words in each category
      vPosMatches <- sum(!is.na(vPosMatches))
      posMatches <- sum(!is.na(posMatches))
      vNegMatches <- sum(!is.na(vNegMatches))
      negMatches <- sum(!is.na(negMatches))
      score <- c(vNegMatches, negMatches, posMatches, vPosMatches)
      #add row to scores table
      newrow <- c(initial_sentence, score)
      final_scores <- rbind(final_scores, newrow)
      
      return(final_scores)
      
    }, vNegTerms, negTerms, posTerms, vPosTerms))
    
    # names of the score column being assigned
    colnames(scores)<-c("X1","vNegTerms_Count","negTerms_Count","posTerms_Count","vPosTerms_Count")
    end_score<-rbind(end_score,scores)
    
  }
  
  
  return(end_score)
}

########
#sentimetal analysis outside loop loading data part

#load up word polarity list and format it
afinn_list <- read.delim(file='E:\\Drexel\\Course-Study\\Fall 2016 Sep19-Dec10\\BUSN-710\\Dows-Project\\AFINN-111.txt', header=FALSE, stringsAsFactors=FALSE)
names(afinn_list) <- c('word', 'score')
afinn_list$word <- tolower(afinn_list$word)

# categorize words as very negative to very positive and add some movie-specific words
# wordlist updation taken from abromberg sentiment_analysis repository
vNegTerms <- afinn_list$word[afinn_list$score==-5 | afinn_list$score==-4]
negTerms <- c(afinn_list$word[afinn_list$score==-3 | afinn_list$score==-2 | afinn_list$score==-1], "second-rate", "moronic", "third-rate", "flawed", "juvenile", "boring", "distasteful", "ordinary", "disgusting", "senseless", "static", "brutal", "confused", "disappointing", "bloody", "silly", "tired", "predictable", "stupid", "uninteresting", "trite", "uneven", "outdated", "dreadful", "bland")
posTerms <- c(afinn_list$word[afinn_list$score==3 | afinn_list$score==2 | afinn_list$score==1], "first-rate", "insightful", "clever", "charming", "comical", "charismatic", "enjoyable", "absorbing", "sensitive", "intriguing", "powerful", "pleasant", "surprising", "thought-provoking", "imaginative", "unpretentious")
vPosTerms <- c(afinn_list$word[afinn_list$score==5 | afinn_list$score==4], "uproarious", "riveting", "fascinating", "dazzling", "legendary")



for(i in 1:nrow(Sample))#
  {
    Records <-sentimentScore(Sample$Text[i], vNegTerms, negTerms, posTerms, vPosTerms)
    Results<-rbind(Results,Records)
    Records<-NA
  
}

for(i in 1:nrow(wipes_reviews))#
{
  Records <-sentimentScore(wipes_reviews$Text[i], vNegTerms, negTerms, posTerms, vPosTerms)
  Results<-rbind(Results,Records)
  Records<-NA
  
}