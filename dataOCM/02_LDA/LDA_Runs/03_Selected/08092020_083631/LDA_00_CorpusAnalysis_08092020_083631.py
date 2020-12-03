# Step 0: Import packages and stopwords
import nltk

if __name__ == '__main__':
    from gensim.models import LdaModel, LdaMulticore
    from gensim import corpora
    import gensim.downloader as api
    from gensim.utils import simple_preprocess, lemmatize
    from nltk.corpus import stopwords
    import re
    import logging
    import csv
    from langdetect import detect
    import spacy
    import xlsxwriter
    import datetime
    from gensim import models
    import shutil
    import os
    import pandas as pd
    import math

    setKeepNounInCorp = True
    setKeepAdjInCorp = True
    setKeepVerbInCorp = True

    nlpDe = spacy.load('de_core_news_sm')

    nlpEn = spacy.load("en_core_web_sm")

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

    def germanSpacyPOS(token):
        return nlpDe(token)[0].pos_
    def englishSpacyPOS(token):
        return nlpEn(token)[0].pos_

    # print(germanSpacyPOS('apple'))
    # print(englishSpacyPOS('language'))

    # print(englishSpacyLemmatizer('discovery'))
    # print(germanSpacyLemmatizer('gehst'))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    stop_words_en = stopwords.words('english')
    stop_words_de = stopwords.words('german')

    dir = 'MasterData_160_companies/'
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    print(files)

    reviews = []
    for f in range(len(files)):
        csvFileName = dir + files[f]
        masterData = list(csv.reader(open(csvFileName, encoding='utf-8'), delimiter='|'))  # CSV file to 2 dimensional list of string
        # reviews.extend([masterData[row][9] for row in range(1,len(masterData))])
        for i in range(1, len(masterData), 10):
            review = masterData[i][9].strip()
            bigReview = ''
            for j in range(i, i + 10):
                bigReview = bigReview + ' ' + masterData[j][9].strip()
            reviews.append(bigReview)
        # for l in reviews:
        #     print(l)

    data_processed = []

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    listNoun = []
    listAdj = []
    listVerb = []
    listNounIds = []
    listAdjIds = []
    listVerbIds = []

    for doc in reviews:
        itsGerman = True
        try:
            if detect(doc) == 'en':
                itsGerman = False
        except:
            itsGerman = True

        doc_out = []

        doc = tokenizer.tokenize(doc)

        if itsGerman == True:

            for wd in doc:
                wd = wd.lower()
                if wd not in stop_words_de:  # remove stopwords
                    # stemmed_word = stemmerDe.stem(wd).lower()  # stemming
                    lemmed_word = germanSpacyLemmatizer(wd)
                    if (germanSpacyPOS(lemmed_word) == 'NOUN' or germanSpacyPOS(lemmed_word) == 'PROPN') and setKeepNounInCorp == True:
                        doc_out = doc_out + [lemmed_word]
                        listNoun.append(lemmed_word)
                    if germanSpacyPOS(lemmed_word) == 'ADJ' and setKeepAdjInCorp == True:
                        doc_out = doc_out + [lemmed_word]
                        listAdj.append(lemmed_word)
                    if germanSpacyPOS(lemmed_word) == 'VERB' and setKeepVerbInCorp == True:
                        doc_out = doc_out + [lemmed_word]
                        listVerb.append(lemmed_word)
                else:
                    continue

        else:

            for wd in doc:
                wd = wd.lower()
                if wd not in stop_words_en:  # remove stopwords
                    # stemmed_word = stemmerDe.stem(wd).lower()  # stemming
                    lemmed_word = englishSpacyLemmatizer(wd)
                    if (englishSpacyPOS(lemmed_word) == 'NOUN' or englishSpacyPOS(lemmed_word) == 'PROPN') and setKeepNounInCorp == True:
                        doc_out = doc_out + [lemmed_word]
                        listNoun.append(lemmed_word)
                    if englishSpacyPOS(lemmed_word) == 'ADJ' and setKeepAdjInCorp == True:
                        doc_out = doc_out + [lemmed_word]
                        listAdj.append(lemmed_word)
                    if englishSpacyPOS(lemmed_word) == 'VERB' and setKeepVerbInCorp == True:
                        doc_out = doc_out + [lemmed_word]
                        listVerb.append(lemmed_word)
                else:
                    continue
        data_processed.append(doc_out)

    listNoun = list(set(listNoun))
    listAdj = list(set(listAdj))
    listVerb = list(set(listVerb))

    dct = corpora.Dictionary(data_processed)
    corpus = [dct.doc2bow(line) for line in data_processed]
    dctMaster = {}
    dctMaster[-1]=['Word','GlobalTF','DF','MaxTFIDF','POS','Entropy','TBD','TBD']
    for key, value in dct.items():
        dctMasterTemp = []
        # dctMasterTemp.append(key)
        dctMasterTemp.append(value)
        dctMasterTemp.append(0)
        dctMasterTemp.append(0)
        dctMasterTemp.append(0)
        if value in listNoun:
            dctMasterTemp.append('NOUN') # POS
        elif value in listAdj:
            dctMasterTemp.append('ADJ') # POS
        elif value in listVerb:
            dctMasterTemp.append('VERB') # POS
        else:
            dctMasterTemp.append('UNDEFINED') # POS
        dctMasterTemp.append(0)
        dctMasterTemp.append(0)
        dctMasterTemp.append(0)
        dctMaster[key]=dctMasterTemp


    tfidf = models.TfidfModel(corpus, id2word=dct)
    dctTfIDF = []
    for bow in corpus:
        for pos in tfidf[bow]:
            if dctMaster[pos[0]][3]<pos[1]:
                dctMaster[pos[0]][3] = pos[1] # MaxTFIDF
        idsBow = set([id for id, qnt in bow])
        for id in idsBow:
            dctMaster[id][2] += 1 # DF
        for pos in bow:
            dctMaster[pos[0]][1] += pos[1] # GlobalTF

    for bow in corpus:
        for pos in bow:
            dctMaster[pos[0]][5] += -(pos[1]/dctMaster[pos[0]][1])*math.log(pos[1]/dctMaster[pos[0]][1],2) # Entropy

    oneByLog2D = 1/math.log(len(corpus),2)
    for key in dctMaster:
        if key!=-1:
            dctMaster[key][5] *= oneByLog2D

    # print(dctMaster)

    workbook = xlsxwriter.Workbook('LDA_00_CorpusAnalysis_dctMaster.xlsx')
    worksheet = workbook.add_worksheet()
    cnt1 = 0
    for key in dctMaster:
        cnt2 = 0
        for val in dctMaster[key]:
            worksheet.write(cnt1,cnt2,val)
            cnt2 += 1
        cnt1 += 1
    workbook.close()

    # took approximately 5 hours to process the kununu reviews of all 160 companies
    # 020-07-22 21:58:05,853 : INFO : built Dictionary(31777 unique tokens: ['flach', 'groß', 'hierarchie', 'vielfalt', 'arbeitgeber']...) from 577130 documents (total 415892 corpus positions)

    # not the average but also the distribution of length of the reviews in master data in each review

    # lot of peaople write little lot write much for example

    # instead of bundling the reviews, use individual reviews


