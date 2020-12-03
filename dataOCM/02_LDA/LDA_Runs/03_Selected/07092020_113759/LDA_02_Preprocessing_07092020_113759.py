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
    import pickle

    setKeepNounInCorp = True
    setKeepAdjInCorp = True
    setKeepVerbInCorp = True

    setTempRun = ''

    nlpDe = spacy.load('de_core_news_sm')

    nlpEn = spacy.load("en_core_web_sm")

    duplicateDictTermsDict = {'arbeitszeiten':'arbeitszeit','arbeiten':'arbeit','nette':'nett','mitarbeitern':'mitarbeiter','interessante':'interessant','teams':'team','neue':'neu','gutes':'gut','abteilungen':'abteilung'}
    def duplicateDictTerms(term):
        if term in duplicateDictTermsDict:
            return duplicateDictTermsDict[term]
        else:
            return term

    def germanSpacyLemmatizer(token):
        token = token.lower()
        lemmed = ''
        for t in nlpDe.tokenizer(token):
            lemmed = lemmed + ' ' + t.lemma_
        term = duplicateDictTerms(lemmed.strip())
        return term

    def englishSpacyLemmatizer(token):
        token = token.lower()
        lemmed = ''
        for t in nlpEn.tokenizer(token):
            lemmed = lemmed + ' ' + t.lemma_
        term = duplicateDictTerms(lemmed.strip())
        return term

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

    csvFileName1 = 'LDA_01_ReviewsPicker_Master_Data_for_training' + setTempRun + '.csv' #Master_Data_Milestone1_Small_for_training.csv'
    masterDataSmall = list(csv.reader(open(csvFileName1, encoding='utf-8'), delimiter='|'))  # CSV file to 2 dimensional list of string
    reviews = [masterDataSmall[row][9] for row in range(1,len(masterDataSmall))]

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

        # doc = nltk.tokenize.word_tokenize(doc)

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

    wordProbs = []
    csvFileName1 = 'LDA_01_ReviewsPicker_keywordsDe.csv'  # Master_Data_Milestone1_Small_for_training.csv'
    impKeywordsDe = list(csv.reader(open(csvFileName1, encoding='utf-8'), delimiter=','))  # CSV file to 2 dimensional list of string
    impKeywordsDeFinal = [i[0] for i in impKeywordsDe]
    csvFileName1 = 'LDA_01_ReviewsPicker_keywordsEn.csv'  # Master_Data_Milestone1_Small_for_training.csv'
    impKeywordsEn = list(csv.reader(open(csvFileName1, encoding='utf-8'), delimiter=','))  # CSV file to 2 dimensional list of string
    impKeywordsEnFinal = [i[0] for i in impKeywordsEn]
    keywordsConstruct1 = [row[1] for row in [row for row in impKeywordsDe+impKeywordsEn if 'overall' == row[0]]]
    keywordsConstruct2 = [row[1] for row in [row for row in impKeywordsDe+impKeywordsEn if 'gender' == row[0]]]
    keywordsConstruct3 = [row[1] for row in [row for row in impKeywordsDe+impKeywordsEn if 'age' == row[0]]]
    keywordsConstruct4 = [row[1] for row in [row for row in impKeywordsDe+impKeywordsEn if 'cultural background' == row[0]]]
    keywordsConstruct5 = [row[1] for row in [row for row in impKeywordsDe+impKeywordsEn if 'sexual orientation' == row[0]]]
    keywordsConstruct6 = [row[1] for row in [row for row in impKeywordsDe+impKeywordsEn if 'handicap' == row[0]]]
    keywordsConstructAll = keywordsConstruct1+keywordsConstruct2+keywordsConstruct3+keywordsConstruct4+keywordsConstruct5+keywordsConstruct6
    keywordsConstructAllIDsInDct = []

    csvFileName = 'LDA_00_CorpusAnalysis_dctMaster.csv'
    dctMaster = list(csv.reader(open(csvFileName, encoding='utf-8'), delimiter=','))  # CSV file to 2 dimensional list of string
    dctWords = [dctMaster[sa][0] for sa in range(1,len(dctMaster))]
    dctWordsIds = []

    keywordsConstructAllNew = []
    keywordsConstructAllIDsInDctNew = []
    listNounNew = []
    listAdjNew = []
    listVerbNew = []
    listNounIdsNew = []
    listAdjIdsNew = []
    listVerbIdsNew = []

    for token, id in dct.token2id.items():
        # print(str(token) + ' ::: ' + str(id))
        if token in keywordsConstructAll:
            keywordsConstructAllIDsInDct.append(id)
        if token in listNoun:
            listNounIds.append(id)
        if token in listAdj:
            listAdjIds.append(id)
        if token in listVerb:
            listVerbIds.append(id)
        if token in dctWords:
            dctWordsIds.append(id)

    dctOpsLog = []
    dctOpsLog.append('Dictionary contains ' + str(len(dct)) + ' terms (Nouns: ' + str(len(listNounIds)) + ' / Adjs: ' + str(len(listAdjIds)) + ' / Verbs: ' + str(len(listVerbIds)) + ' / ImpKeywords: ' + str(len(keywordsConstructAllIDsInDct)) + ') before filtering out bad terms.')
    print(dctOpsLog[-1])
    dctOpsLog.append('Filtering the dictionary to keep only the important terms...')
    print(dctOpsLog[-1])

    dct.filter_tokens(good_ids=list(dctWordsIds))
    finalNosImpKeywords = 0
    finalNosNouns = 0
    finalNosAdjs = 0
    finalNosVerbs = 0

    for token, id in dct.token2id.items():
        if token in keywordsConstructAll:
            finalNosImpKeywords = finalNosImpKeywords + 1
            keywordsConstructAllNew.append(token)
            keywordsConstructAllIDsInDctNew.append((id))
        if token in listNoun:
            finalNosNouns = finalNosNouns + 1
            listNounNew.append(token)
            listNounIdsNew.append(id)
        if token in listAdj:
            finalNosAdjs = finalNosAdjs + 1
            listAdjNew.append(token)
            listAdjIdsNew.append(id)
        if token in listVerb:
            finalNosVerbs = finalNosVerbs + 1
            listVerbNew.append(token)
            listVerbIdsNew.append(id)
    dctOpsLog.append('Dictionary contains ' + str(len(dct)) + ' terms (Nouns: ' + str(finalNosNouns) + ' / Adjs: ' + str(finalNosAdjs) + ' / Verbs: ' + str(finalNosVerbs) + ' / ImpKeywords: ' + str(finalNosImpKeywords) + ') after filtering out bad terms.')
    print(dctOpsLog[-1])

    corpus = [dct.doc2bow(line) for line in data_processed]

    dct.save('LDA_02_Preprocessing_Dictionary.dictionary')
    pickle.dump(corpus, open('LDA_02_Preprocessing_Corpus.corpus', 'wb'))
    pickle.dump(keywordsConstructAllNew, open('LDA_02_Preprocessing_keywordsConstructAllNew.list', 'wb'))
    pickle.dump(keywordsConstructAllIDsInDctNew, open('LDA_02_Preprocessing_keywordsConstructAllIDsInDctNew.list', 'wb'))
    pickle.dump(listNounNew, open('LDA_02_Preprocessing_listNounNew.list', 'wb'))
    pickle.dump(listAdjNew, open('LDA_02_Preprocessing_listAdjNew.list', 'wb'))
    pickle.dump(listVerbNew, open('LDA_02_Preprocessing_listVerbNew.list', 'wb'))
    pickle.dump(listNounIdsNew, open('LDA_02_Preprocessing_listNounIdsNew.list', 'wb'))
    pickle.dump(listAdjIdsNew, open('LDA_02_Preprocessing_listAdjIdsNew.list', 'wb'))
    pickle.dump(listVerbIdsNew, open('LDA_02_Preprocessing_listVerbIdsNew.list', 'wb'))
    pickle.dump(dctOpsLog, open('LDA_02_Preprocessing_dctOpsLog.list', 'wb'))

# takes max 5 hours