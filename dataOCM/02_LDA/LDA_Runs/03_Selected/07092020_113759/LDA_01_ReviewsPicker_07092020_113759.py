import csv
import os

from langdetect import detect
import spacy
import nltk
from nltk.corpus import stopwords

csvFileName = 'LDA_00_CorpusAnalysis_KeywordsTable_output_IMP.csv'
keywordsTable = list(csv.reader(open(csvFileName,encoding='utf-8'),delimiter=',')) # CSV file to 2 dimensional list of string
csvFileName = 'LDA_00_CorpusAnalysis_dctMaster.csv'
dctMaster = list(csv.reader(open(csvFileName,encoding='utf-8'),delimiter=',')) # CSV file to 2 dimensional list of string
dctWords = [dctMaster[i][0] for i in range(1,len(dctMaster))] # if len(dctMaster[i][6]) == 2]
print(dctWords)
keywordsTableDe = []
keywordsTableEn = []

nlpDe = spacy.load('de_core_news_sm')
nlpEn = spacy.load("en_core_web_sm")

stop_words_en = stopwords.words('english')
stop_words_de = stopwords.words('german')

tokenizer = nltk.RegexpTokenizer(r"\w+")

def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

def germanSpacyLemmatizer(token):
    token = token.lower()
    lemmed = ''
    for t in nlpDe.tokenizer(token):
        lemmed = lemmed + ' ' + t.lemma_
    return lemmed.strip()

def englishSpacyLemmatizer(token):
    token = token.lower()
    lemmed = ''
    for t in nlpEn.tokenizer(token):
        lemmed = lemmed + ' ' + t.lemma_
    return lemmed.strip()

keywordsOutDe = []
keywordsOutEn = []
for j in range(len(keywordsTable)):
    for k in range(1,len(keywordsTable[j])):
        keyword = keywordsTable[j][k]
        keywordLang = keyword[0:3]
        keyword = keyword.replace('en:','').replace('de:','')
        itsGermanKeyword = True
        if keywordLang == "en:":
            itsGermanKeyword = False
        else:
            itsGermanKeyword = True
        if itsGermanKeyword == True:
            keyword = germanSpacyLemmatizer(keyword)
            keywordsTableDe.append(keyword)
            keywordsOutDe_temp = []
            keywordsOutDe_temp.append(keywordsTable[j][0])
            keywordsOutDe_temp.append(keyword)
            keywordsOutDe.append(keywordsOutDe_temp)
        else:
            keyword = englishSpacyLemmatizer(keyword)
            keywordsTableEn.append(keyword)
            keywordsOutEn_temp = []
            keywordsOutEn_temp.append(keywordsTable[j][0])
            keywordsOutEn_temp.append(keyword)
            keywordsOutEn.append(keywordsOutEn_temp)

keywordsTableEn = unique(keywordsTableEn)
keywordsTableDe = unique(keywordsTableDe)
csvFileNameOut = 'LDA_01_ReviewsPicker_keywordsEn.csv'
csvFileOut = open(csvFileNameOut, "w", newline='', encoding='utf-8')
csv_out = csv.writer(csvFileOut, delimiter=',')
for c in range(len(keywordsOutEn)):
    csv_out.writerow(keywordsOutEn[c]) # + features)
csvFileNameOut = 'LDA_01_ReviewsPicker_keywordsDe.csv'
csvFileOut = open(csvFileNameOut, "w", newline='', encoding='utf-8')
csv_out = csv.writer(csvFileOut, delimiter=',')
for c in range(len(keywordsOutDe)):
    csv_out.writerow(keywordsOutDe[c]) # + features)

print('Keywords files created.')

def reviewHit(review):
    fetchThis = False
    doc = review
    itsGerman = True
    try:
        if detect(doc) == 'en':
            itsGerman = False
    except:
        itsGerman = True

    doc = tokenizer.tokenize(doc)

    if itsGerman == True:
        for wd in doc:
            wd = wd.lower()
            if wd not in stop_words_de:  # remove stopwords
                # stemmed_word = stemmerDe.stem(wd).lower()  # stemming
                lemmed_word = germanSpacyLemmatizer(wd)
                if lemmed_word in dctWords:
                    fetchThis = True
                    return fetchThis
            else:
                continue
    else:
        for wd in doc:
            wd = wd.lower()
            if wd not in stop_words_en:  # remove stopwords
                # stemmed_word = stemmerDe.stem(wd).lower()  # stemming
                lemmed_word = englishSpacyLemmatizer(wd)
                if lemmed_word in dctWords:
                    fetchThis = True
                    return fetchThis
            else:
                continue

csvFileNameOut = 'LDA_01_ReviewsPicker_Master_Data_for_training.csv'
csvFileOut = open(csvFileNameOut, "w", newline='', encoding='utf-8')
csv_out = csv.writer(csvFileOut, delimiter='|')

dir = 'MasterData_160_companies/'
files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
print(files)

firstFile = True
totalReviews = 0
for f in range(len(files)):
    csvFileName = dir + files[f]
    masterData = list(csv.reader(open(csvFileName, encoding='utf-8'), delimiter='|'))  # CSV file to 2 dimensional list of string

    if firstFile:
        csv_out.writerow(masterData[0]) # + features)
        firstFile = False

    # for i in range(1,len(masterData)):
    #     review = masterData[i][9].strip()
    #     # if (review != ''):
    #     #     csv_out.writerow(masterData[i])
    #     # if ((masterData[i][7] == 'Gleichberechtigung' or masterData[i][7] ==  'Umgang mit älteren Kollegen') and review != '') or reviewHit(review) == True:
    #     #     csv_out.writerow(masterData[i])
    #     if reviewHit(review) == True and review != '':
    #         csv_out.writerow(masterData[i])
    #
    #     if i%100 == 0:
    #         print(str(i) + " reviews processed.")

    for i in range(1,len(masterData),10):
        review = masterData[i][9].strip()
        bigReview = ''
        for j in range(i,i+10):
            bigReview = bigReview + ' ' + masterData[j][9].strip()
        masterData[i][9] = bigReview
        if (len(bigReview.strip()) > 50 and reviewHit(bigReview) == True):
            csv_out.writerow(masterData[i])
            totalReviews += 1
        # if ((masterData[i][7] == 'Gleichberechtigung' or masterData[i][7] ==  'Umgang mit älteren Kollegen') and review != '') or reviewHit(review) == True:
        #     csv_out.writerow(masterData[i])
        # if reviewHit(review) == True:
        #     csv_out.writerow(masterData[i])

        if i%100 == 0:
            print(str(i) + " reviews processed.")

print('total reviews are', str(totalReviews))

# csvFileNameOut = 'pickedReviews.csv'
# csvFileOut = open(csvFileNameOut, "w", newline='', encoding='utf-8')
# csv_out = csv.writer(csvFileOut, delimiter='|')
# csv_out.writerow(masterData[0][7:10]) # + features)

# takes approx half hour to process 160 companies reviews