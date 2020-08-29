import json
from nltk import sent_tokenize, corpus
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora, utils, models
from googleapiclient.discovery import build
import csv
import nltk
import twitter
import Keys
import spacy
import re
import math


class FakeNewsNetItem:
    #Class to hold a document's attributes, for the FakeNewsNet dataset
    text = ""
    truthfulness = 0


#Dataset file paths
DATASETS_PATH = "E:/Documents/fake_news/Datasets/"
DATASET_PATH_FAKENEWSNET_DATASET = DATASETS_PATH + "FakeNewsNet"
DATASET_PATH_IBM_DATASET = DATASETS_PATH + "IBM_Debater_(R)_CE-EMNLP-2015.v3"
DATASET_PATH_SENTIMENT140_DATASET = DATASETS_PATH + "Sentiment140/training.1600000.processed.noemoticon.csv"

#Declare and define important global variables used throughout the program
dictionary = corpora.Dictionary()
csv.field_size_limit(1000000000)
spacy.load('en_core_web_sm')
STOPWORDS = set(corpus.stopwords.words("english")).union(spacy.lang.en.stop_words.STOP_WORDS)
TWO_DP_FORMAT = "%.2f"
model = None


def twitterContextSearch(query):
    #Method to calculate the support score for a query on Twitter.
    #'query' - a single string to search for
    #Returns the support score, 0 if no results could be found or support is neutral

    api = twitter.Api(consumer_key=Keys.TWITTER_CONSUMER_KEY,
                  consumer_secret=Keys.TWITTER_CONSUMER_SECRET,
                  access_token_key=Keys.TWITTER_ACCESS_TOKEN_KEY,
                  access_token_secret=Keys.TWITTER_ACCESS_TOKEN_SECRET)

    #Form a search query and run with the API
    raw_query = "q=" + query + "%20&result_type=recent&count=10&lang=en"
    results = api.GetSearch(raw_query=raw_query)

    #Perform sentiment analysis on the returned tweets using SentimentIntensityAnalyzer
    #Use the retweet and favourite counts to calculate support for the tweet
    #If there is no clear detected leaning in sentiment, ignore as likely to be unreliable

    analyser = SentimentIntensityAnalyzer()
    support = 0
    for tweet in results:
        scores = analyser.polarity_scores(tweet.text)
        if abs(scores["compound"]) > 0.3:
            #Use logartihm here to reduce the effect of certain tweets having dominating levels of support
            support += math.log((tweet.retweet_count + tweet.favorite_count*2 + 0.001)) * scores["compound"]

    return support


#https://support.google.com/customsearch/answer/4513886?visit_id=637177890572977240-812858338&rd=1
def googleSearch(search_service, search_query):
    #Method to perform a single search of Google, returning the list of results
    #search_service - the Google search service object to use
    #search_query - the query to search, a single string
    #Returns an empty list if no results found

    res = search_service.cse().list(q=search_query, cx=Keys.GOOGLE_CUSTOM_SEARCH_ENGINE_ID, num=10).execute()
    if int(res["searchInformation"]["totalResults"]) > 0:
        return res['items']
    return []


def preProcess(string):
    #Pre-process the given string, removing stopwords, small words, and tokenising into a list of words
    #Returns a list of strings (words)

    return [word for word in utils.simple_preprocess(string, min_len=5) if word not in STOPWORDS]


def webVerify(search_service, query, known_domains):
    #Verify the given query using Google Search
    #search_service - the Google search service object to use
    #search_query - the query to verify, a single string
    #known_domains - the parsed CSV file of specified domains
    #Returns the support score for the query, 0 if neutral/mixed

    #Perform a Google search on the query
    items = googleSearch(search_service, query)

    #For each search result
    support = 0
    for item in items:
        source_domain = item["displayLink"].lower()

        #Find the first (if any) domain in the list
        first = next((x for x in known_domains if source_domain.endswith(x["domain"])), None)

        #Adjust the support value by the reliability measure for the domain
        if first != None:
            support += int(first["reliability"])
    return support


def getFile(file_path):
    #Load the file located at file_path
    #Returns the text contents of the file, or None if an error occured
    try:
        with open(file_path, "r", encoding="utf8") as file:
            content = file.read()
            #File closed automatically by the with statement
    except FileNotFoundError:
        content = None
        #print("Error loading " + file_path)
    return content


def parseIBMDataset():
    #Load in and parse the dataset
    print("Loading IBM dataset...", end="")
    lst = []
    for record in parseCSVFile(DATASET_PATH_IBM_DATASET + "/articles.txt", "\t"):
        lst.append(getFile(DATASET_PATH_IBM_DATASET + "/articles/clean_" + record["article Id"] + ".txt"))
    print("done.")
    return lst


def parseFakeNewsNetCSVFile(type):
    #Load in and parse one of the CSV files from the FakeNewsNet dataset
    lst = []
    truthfulness = 1 if type == "real" else -1
    for record in parseCSVFile(DATASET_PATH_FAKENEWSNET_DATASET + "/dataset/politifact_" + type + ".csv", ","):
        path = DATASET_PATH_FAKENEWSNET_DATASET + "/code/fakenewsnet_dataset/politifact/" + type +"/" + record["id"] + "/"
        content = parseJSONFile(path + "news content.json")
        if content == None:
            continue
        item = FakeNewsNetItem()
        item.text = content["text"]
        item.truthfulness = truthfulness
        lst.append(item)
    return lst


def parseFakeNewsNetDataset():
    #Load in and parse the FakeNewsNet dataset
    print("Loading FakeNewsNet dataset...", end="")
    items = parseFakeNewsNetCSVFile("fake")
    items.extend(parseFakeNewsNetCSVFile("real"))
    print("done.")
    return items


def parseSentiment140Dataset():
    #Load in and parse the Sentiment140 dataset
    print("Loading Sentiment140 dataset...", end="")
    items = parseCSVFile(DATASET_PATH_SENTIMENT140_DATASET, ",", ["Sentiment", "ID", "Date", "Flag", "User", "Text"], "ISO-8859-1")
    print("done.")
    return items


def parseCSVFile(path, delimiter, fieldnames=None, encoding="utf-8-sig"):
    #Load in and parse a CSV file
    #path - the filepath at which it is located
    #delimiter - the separator type used in the CSV file
    #fieldnames - a list of strings representing the headings of the CSV file, in order left-to-right
    #Returns the parsed CSV file
    lst = []
    with open(path, "r", encoding=encoding) as file:
        reader = csv.DictReader(file, delimiter=delimiter, fieldnames=fieldnames)
        for line in reader:
            lst.append(line)
    return lst


def parseJSONFile(file_path):
    #Load in and parse a JSON file
    #file_path - the filepath at which it is located
    #Returns the parsed JSON file
    content = getFile(file_path)
    if content != None:
        parsed = json.loads(content)
        return parsed


def extractPremisesAndConclusions(sentences, discourse_indicators):
    #Extract the premises and conclusions from a list of documents
    #documents - a list of sentences (strings)
    #discourse_indicators - the parsed CSV file of discourse indicators
    #Returns a list in format [list of premise strings, list of conclusion strings]
    premises = []
    conclusions = []
    new_documents = sentences[:]
    pop_count = 0
    for i, sentence in enumerate(new_documents):

        #Attempt to find a discourse indicator in the sentence
        first = next((word for word in discourse_indicators if word["word"] in sentence), None)

        if first != None:

            delimiter = first["word"]
            index = sentence.find(delimiter)
            last_part = sentence[index+len(delimiter)+1:].replace(".", "").strip()
            first_part = sentence[:index].strip()

            if len(first_part) < 60 or len(last_part) < 60:
                continue

            #In the discourse_indicators file, the 'relation' column defines whether the premise/conclusion is before or after the indicator
            relation = int(first["relation"])
            if relation==0:
                premise = first_part #Premise was last
                conclusion = last_part
            else:
                premise = last_part #Premise was first
                conclusion = first_part

            premises.append(premise)
            sentences.pop(i-pop_count)
            pop_count+=1
            #If a "conclusion" contains digits, it is more likely a premise (containing statistics)
            if any(word.isdigit() for word in conclusion):
                premises.append(conclusion)
            else:
                conclusions.append(conclusion)

    return [premises, conclusions]


def bayesPrecisionTesting(pc_pairs, model):
    #Test the trained Bayes model for determining if a string is a premise / conclusion
    #pc_pairs - the list of correct answers, in format [list of premise strings, list of conclusion strings]
    #model - the trained Bayes model
    print("Testing trained Bayes model on annotated data...")
    number = 0
    true_positives = true_negatives = false_positives = false_negatives = 0
    for index, part in enumerate(pc_pairs):
        for sentence in part:
            number+=1
            result =  model.classify(dict(dictionary.doc2bow(sentence.split())))
            true_label = "premise" if index == 0 else "conclusion"
            if result=="premise":
                if true_label=="premise":
                    true_positives+=1
                else:
                    false_positives+=1
            elif result=="conclusion":
                if true_label=="conclusion":
                    true_negatives+=1
                else:
                    false_negatives+=1

    print("Premises:")
    premise_data = printF1(true_positives, false_positives, false_negatives)
    print("Conclusions:")
    conclusion_data = printF1(true_negatives, false_negatives, false_positives)

    with open(str(i) + "_bayes.csv", "w", newline="") as file:
        fieldnames = ["Threshold", "premise precision", "premise recall", "premise F1", "conclusion precision", "conclusion recall", "conclusion F1"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict(zip(fieldnames, [threshold, premise_data["Precision"], premise_data["Recall"], premise_data["F1"], conclusion_data["Precision"], conclusion_data["Recall"], conclusion_data["F1"]])))

    print("done.")


def topicModellingFilter(tokenised_sentences, lda, bowcorpus):
    #Removes sentences from tokenised_sentences with low topic occurrence
    #tokenised_sentences - the list of sentences in string format
    #lda - the trained LDA model
    #bowcorpus - a list of sentences in bag-of-words format
    tokenised_words = [preProcess(sentence) for sentence in tokenised_sentences]
    #Retrieve topics for the documents from the LDA model
    topics = [lda[doc] for doc in bowcorpus] #Results vary slightly due to LDA randomness
    offset = 0
    for index, text in enumerate(topics):
        vals_set = [topic[1] for topic in text]
        if any(text) and (max(vals_set)-min(vals_set)) >= threshold:
            tokenised_sentences.pop(index-offset)
            offset+=1
    return tokenised_sentences


def compareFindings(found_list, test_list):
    #Returns the number of strings in found_list which are substrings of the strings in test_list
    #Used to measure the recall rates of found strings (found_list) from the original (test_list)
    #found_list - the list of strings representing those extracted from the documents
    #test_list - the list of sentences loaded from the dataset

    count = 0
    for found in found_list:
        for test in test_list:
            if found in test:
                count+=1
                break
    return count


def removeTrainingData(found_list, test_list):
    #Removes the strings in found_list from test_list
    #This is to ensure the separation of the training and testing sets,
    #when the extracted strings are being trained and tested on the same dataset 
    #found_list - the list of strings extracted computationally from the dataset
    #test_list - the list of original document strings
    return_list = []
    for found in found_list:
        for test in test_list:
            if found in test:
                return_list.append(test)
    return [x for x in test_list if x not in return_list]


def testSentiment():
    #Perform a test of NLTK's sentiment analysis on a dataset of tweets

    data = parseSentiment140Dataset()

    print("Testing NLTK sentiment analysis...", end="")
    analyser = SentimentIntensityAnalyzer()
    true_positives = false_positives = true_negatives = false_negatives = 0
    for tweet in data:
        scores = analyser.polarity_scores(tweet["Text"])
        if int(tweet["Sentiment"]) == 0:
            if int(scores["compound"] > 0):
                false_positives+=1
            else:
                true_negatives+=1
        else:          
            if int(scores["compound"] > 0):
                true_positives+=1
            else:
                false_negatives+=1
        #tweet["Sentiment"] #0=negative, 4=positive
    printF1(true_positives, false_positives, false_negatives)



def zeroDiv(numerator, denominator):
    #Special version of divide that returns 0 if denominator is 0
    return 0 if denominator==0 else numerator / denominator


def printF1(true_positives, false_positives, false_negatives):
    #Print the precision, recall and F1 scores of the given data
    #true_positives - the number of true positives
    #false_positives - the number of false positives
    #false_negatives - the number of false negatives
    #Returns a dictionary with the calculated values
    precision = zeroDiv(true_positives, true_positives+false_positives)
    recall = zeroDiv(true_positives, true_positives+false_negatives)
    f1_score = zeroDiv(2 * precision * recall, precision + recall)
    print(TWO_DP_FORMAT % precision + " precision")
    print(TWO_DP_FORMAT % recall + " recall")
    print(TWO_DP_FORMAT % f1_score + " F1")

    return {"Precision" : precision, "Recall" : recall, "F1" : f1_score}


def testArgumentMining(model, pc_pairs, same_test_train_data):
    #Run tests on the argument mining performance
    #model - the trained premise/conclusion classifier
    #pc_pairs - the extracted list of [list of extracted premises, list of extracted conclusions]
    #same_test_train_data - bool, if true tests recall

    print("Preparing tests...", end="")
    testing_premises = list(set([re.sub("\[REF$", "[REF]", record["Premise"]) for record in parseCSVFile(DATASET_PATH_IBM_DATASET + "/evidence.txt", "\t", ["Topic", "Claim", "Premise"])]))
    testing_conclusions = list(set([record["Claim corrected version"] for record in parseCSVFile(DATASET_PATH_IBM_DATASET + "/claims.txt", "\t")]))
    found_premises = [sentence for pair in pc_pairs for sentence in pair[0]]
    found_conclusions = [sentence for pair in pc_pairs for sentence in pair[1]]
    print("done.")

    #Test recall
    if same_test_train_data:

        print("\nTesting detected premises and conclusions...")
        true_premises = compareFindings(found_premises, testing_premises)
        false_premises = compareFindings(found_premises, testing_conclusions)
        true_conclusions = compareFindings(found_conclusions, testing_conclusions)
        false_conclusions = compareFindings(found_conclusions, testing_premises)

        print(str(len(found_premises)) + " premises found")
        premise_data = printF1(true_premises, false_premises, false_conclusions)
        print("")
        print(str(len(found_conclusions)) + " conclusions found")
        conclusion_data = printF1(true_conclusions, false_conclusions, false_premises)
        print("")

        with open(str(i) + "_extracted.csv", "w", newline="") as file:
            fieldnames = ["Threshold", "premises found", "premise precision", "premise recall", "premise F1", "conclusions found", "conclusion precision", "conclusion recall", "conclusion F1"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(dict(zip(fieldnames, [threshold, len(found_premises), premise_data["Precision"], premise_data["Recall"], premise_data["F1"], len(found_conclusions), conclusion_data["Precision"], conclusion_data["Recall"], conclusion_data["F1"]])))


    #Test precision
    print("Testing undetected premises and conclusions...")
    #Only a subset of the premises and conclusions are able to be found computationality.
    #These are removed from the testing set to ensure that the testing and training sets are disjoint.
    testing_premises = removeTrainingData(found_premises, testing_premises)
    testing_conclusions = removeTrainingData(found_conclusions, testing_conclusions)

    #Test the trained model
    bayesPrecisionTesting([testing_premises, testing_conclusions], model)


def argumentMine(training_documents):
    #Perform argument mining
    #training_documents - a list of strings (documents)
    #Returns extracted pc_pairs, trained classifier model

    ###########################################################################################
    #Step1: Load in the data
    ###########################################################################################
    print("Loading files...", end="")
    discourse_indicators = parseCSVFile("discourse_indicators.csv", ",")
    tokenised_sentences = [sent_tokenize(document) for document in training_documents]
    tokenised_words = [[preProcess(sentence) for sentence in document] for document in tokenised_sentences]
    combined_corpus = [sentence for document in tokenised_words for sentence in document]
    dictionary.add_documents(combined_corpus)
    print("done.")

    ###########################################################################################
    #Step2: Filter out non-argumentative sentences via topic modelling techniques
    ###########################################################################################
    print("Topic modelling to filter argumentative units...", end="")

    doccorpus = [[dictionary.doc2bow(word) for word in sentence] for sentence in tokenised_words ]
    bowcorpus = [sentence for doc in doccorpus for sentence in doc] #Document to bag of words - returns the word counts   
    temp = dictionary[0] #Needed to populate id2token attribute
    lda = models.LdaModel(corpus=bowcorpus, id2word=dictionary.id2token, num_topics=100)   
    
    iterator = iter(doccorpus)
    for index, document in enumerate(tokenised_sentences):
        tokenised_sentences[index] = topicModellingFilter(document, lda, next(iterator))
    print("done.")

    ###########################################################################################
    #Step3: Split arguments into premises and conclusions
    ###########################################################################################
    print("Extracting premise-conclusion pairs...", end="")
    pc_pairs = []
    for index in range(len(tokenised_sentences)):
        pc_pairs.append(extractPremisesAndConclusions(tokenised_sentences[index], discourse_indicators)) #Extracts and removes used sentences from tokenised_sentences 
    print("done.")

    ###########################################################################################
    #Step4: Use machine learning to extract more premises at sentence level granularity
    ###########################################################################################
    model = learn(pc_pairs)

    for index, document in enumerate(tokenised_sentences):        
        for sentence in document:           
            part = model.classify(dict(dictionary.doc2bow(sentence.split())))
            pc_pairs[index][0 if part=="premise" else 1].append(sentence)                          

    return pc_pairs, model


def learn(data):
    #Train the classifier on the data
    #data - in form of a list of documents, where each is a list of [premise strings, conclusion strings]
    #Returns the trained classifier

    print("Training Naive Bayes Classifier for finding premises...", end="")

    premises = [sentence for document in data for sentence in document[0]]
    conclusions = [sentence for document in data for sentence in document[1]]

    #Ensure that training set has balanced class sizes
    size = min([len(premises), len(conclusions)])
    premises = premises[:size]
    conclusions = conclusions[:size]

    #Ensure sentence words are added to the dictionary
    dictionary.add_documents([sentence.split() for sentence in premises])
    dictionary.add_documents([sentence.split() for sentence in conclusions])
    
    x = [(dict(dictionary.doc2bow(y.split())), "premise") for y in premises]
    x.extend([(dict(dictionary.doc2bow(y.split())), "conclusion") for y in conclusions])

    classifier = nltk.NaiveBayesClassifier.train(x)

    print("done.") 
    return classifier


def verify(premises):
    #Verify the truthfulness of premises via APIs
    #premises - a list of strings (premises)
    #Returns a list of verdicts where -1 is unverified, 1 is verified
    known_domains = parseCSVFile("domains.csv", ",")
    search_service = build("customsearch", "v1", developerKey=Keys.GOOGLE_SEARCH_API_KEY)
    document_verdicts = []
    google_query_count = 0
    for index, document in enumerate(premises):
        web_support = twitter_support =  0
        for premise in document:
            words = preProcess(premise)
            if len(words) > 1:
                query = " ".join(words)
                dictionary.add_documents([words])

                if len(words) < 6:
                    print("Verifying Twitter support...", end="")
                    twitter_support += twitterContextSearch(query)
                    print("done.")

                elif len(words) < 10:
                    print("Verifying via Google...", end="")      
                    if google_query_count==100:
                        return document_verdicts  
                    google_query_count+=1
                    web_support += webVerify(search_service, query, known_domains) #"Donald Trump"           
                    print("done.")

        support = web_support + twitter_support
        verdict = -1 if support < 0 else 1
        if support == 0: #If not enough evidence, mark as unknown
            verdict = 0
        print(str(verdict) + ", " + str(twitter_support) + ", " + str(web_support))
        document_verdicts.append(verdict)

    return document_verdicts


def testVerification(verified, solution):
    #Test the precision of the article truthfulness verdicts
    #verified - list of calculated article verdicts 
    #solution - the true article verdicts
    true_positives = true_negatives = false_positives = false_negatives = unverified = 0
    iterator = iter(solution)
    for pred_val in verified:
        true_val = next(iterator)
        if pred_val == -1:
            if true_val == -1:
                true_positives+=1
            else:
                false_positives+=1
        elif pred_val == 1:
            if true_val == -1:
                false_negatives+=1
            else:
                true_negatives+=1
        else:
            unverified+=1

    print(str(len(verified)) + " total articles")
    printF1(true_positives, false_positives, false_negatives)
    print(str(unverified) + " articles could not have truthfulness predicted")


print("Arbiter News Verification System")

while(True):
    #Begin program execution with mode selection
    selection = input("Please choose a function:\n(1) Test argument mining on IBM dataset\n(2) Test verification on FakeNewsNet dataset\n(3) Test sentiment analysis on Sentiment140 dataset\n\n")

    if selection=="1":
        #Argument Mining
        articles = parseIBMDataset()

        for i in range(1, 10):
            threshold = i*0.1
            pc_pairs, model = argumentMine(articles)
            #Test premise and conclusion extraction model
            testArgumentMining(model, pc_pairs, True)         


    elif selection=="2":
        #Argument Mining
        dataset = parseFakeNewsNetDataset()
        articles = [article.text for article in dataset]
        threshold = 0.3
        pc_pairs, model = argumentMine(articles)

        #Verification
        verified = verify([pair[0] for pair in pc_pairs])
        solution = [document.truthfulness for document in dataset]

        #Test verification
        testVerification(verified, solution)


    elif selection=="3":
        #Test Sentiment Analysis
        testSentiment()


    else:
        print("Invalid input.\n")




