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

    currDateTime = '07092020_113759' # Dataset1
    # noOfTopicsFolder = 'noOfTopics_22'  # Dataset1: 22 # selected by team # given to Sarah
    noOfTopicsFolder = 'noOfTopics_18'  # Dataset1: 18 # given to Sarah
    # noOfTopicsFolder = 'noOfTopics_20'  # Dataset1: 20 # given to Sarah

    # currDateTime = '08092020_083631'  # Dataset2
    # noOfTopicsFolder = 'noOfTopics_23' # Dataset2: 23 # selected by team
    # noOfTopicsFolder = 'noOfTopics_19'  # Dataset1: 19 # given to Sarah
    # noOfTopicsFolder = 'noOfTopics_21'  # Dataset1: 21 # given to Sarah

    dir = 'LDA_Runs/03_Selected/' + currDateTime

    setFitBundle = False

    setTempRun = '' #change benchmark review size also

    nlpDe = spacy.load('de_core_news_sm')

    nlpEn = spacy.load("en_core_web_sm")

    tokenizer = nltk.RegexpTokenizer(r"\w+")

    stop_words_en = stopwords.words('english')
    stop_words_de = stopwords.words('german')

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

    # load the model
    lda_model = LdaModel.load(dir + '/' + noOfTopicsFolder + '/' + 'LDA_03_Training_Model_' + currDateTime + '_' + noOfTopicsFolder + '.model')
    noOfTopics = lda_model.num_topics

    dct = corpora.dictionary.Dictionary.load(dir + '/' + 'LDA_02_Preprocessing_Dictionary_' + currDateTime +'.dictionary')

    # See the topics
    lda_model.print_topics(-1)
    # print(lda_model.get_topic_terms(0,370))
    # print(lda_model.get_topic_terms(1,370))
    # print(lda_model.get_term_topics(102,1))

    # dir = 'MasterData_160_companies/'
    # files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    # print(files)

    benchmarkReviews = []
    fittedData = []
    # for f in range(len(files)):
    csvFileName = dir + '/' + 'LDA_01_ReviewsPicker_Master_Data_for_training_' + currDateTime +'.csv'
    masterDataBig = list(csv.reader(open(csvFileName, encoding='utf-8'), delimiter='|'))  # CSV file to 2 dimensional list of string

    csvFileNameOut = dir + '/' + noOfTopicsFolder + '/' + 'LDA_04_Fitting_Master_Data_fitted_' + currDateTime +'.csv'
    csvFileOut = open(csvFileNameOut, "w", newline='', encoding='utf-8')
    csv_out = csv.writer(csvFileOut, delimiter='|')

    csv_out.writerow(masterDataBig[0] + ['topic' + str(i) for i in range(noOfTopics)])
    fittedData.append(masterDataBig[0] + ['topic' + str(i) for i in range(noOfTopics)])
    # for j in range(108, 110):  # len(masterDataBig)):

    loopStep = 1
    if setFitBundle == True:
        loopStep = 10
    for j in range(1, len(masterDataBig),loopStep):
        doc = masterDataBig[j][9].strip()
        if setFitBundle == True:
            doc = ''
            for k in range(j, j + 10):
                doc = doc + ' ' + masterDataBig[k][9].strip()
            masterDataBig[j][9] = doc

        if len(doc) > 5:

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
                        if lemmed_word:
                            doc_out = doc_out + [lemmed_word]
                    else:
                        continue

            else:

                for wd in doc:
                    wd = wd.lower()
                    if wd not in stop_words_en:  # remove stopwords
                        # stemmed_word = stemmerDe.stem(wd).lower()  # stemming
                        lemmed_word = englishSpacyLemmatizer(wd)
                        if lemmed_word:
                            doc_out = doc_out + [lemmed_word]
                    else:
                        continue

            corpus2 = [dct.doc2bow(doc_out)]
            # print('============ corpus 2 =========')
            # print(corpus2)
            # word_counts = [[(dct[id], count) for id, count in line] for line in corpus2]
            # print(word_counts)
            vector = lda_model[corpus2[0]]  # get topic probability distribution for a document
            # vector = lda_model[lda_model.id2word.doc2bow(doc_out)]  # get topic probability distribution for a document
            # print(vector)
            vector2 = vector[0]
            finalVector = []
            for k in range(noOfTopics):
                finalVector_temp = []
                finalVector_temp.append(k)
                finalVector_temp.append(0)
                for l in range(len(vector2)):
                    if vector2[l][0] == k:
                        finalVector_temp[1] = vector2[l][1]
                finalVector.append(finalVector_temp)
            csv_out.writerow(masterDataBig[j] + [row[1] for row in finalVector])
            fittedData.append(masterDataBig[j] + [row[1] for row in finalVector])
            if len(masterDataBig[j][9].strip()) > 1:
                benchmarkReviewsTemp = []
                benchmarkReviewsTemp.append(masterDataBig[j][7].strip())
                benchmarkReviewsTemp.append(masterDataBig[j][8].strip())
                benchmarkReviewsTemp.append(masterDataBig[j][9].strip())
                benchmarkReviewsTemp.append([row[1] for row in finalVector])
                benchmarkReviews.append(benchmarkReviewsTemp)
        if j % 100 == 0:
            print(str(j) + " reviews processed.")

    def reportIt():
        # newpath = r'C:\Program Files\arbitrary'
        # if not os.path.exists(newpath):
        #     os.makedirs(newpath)
        # now = datetime.datetime.now()
        # currDateTime = (now.strftime("%d%m%Y_%H%M%S"))
        # dir = pickle.load(open('LDA_03_Training_LastRunTime.string', 'rb'))
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy2('LDA_04_Fitting.py', dir + '/' + noOfTopicsFolder + '/' +  'LDA_04_Fitting_' + currDateTime + '.py')
        # pickle.dump(fittedData, open(dir + '/LDA_04_Fitting_fittedData_' + currDateTime + '.data'))
        with open(dir + '/' + noOfTopicsFolder + '/' +  'LDA_04_Fitting_fittedData_' + currDateTime + '.data', 'wb') as f:
            pickle.dump(fittedData, f)
        workbook = xlsxwriter.Workbook(dir + '/' + noOfTopicsFolder + '/' + 'LDA_04_Fitting_Report_' + currDateTime + '.xlsx')
        worksheet2 = workbook.add_worksheet()

        formatBold = workbook.add_format()
        formatBold.set_bold()
        formatRedLeft = workbook.add_format()
        formatRedLeft.set_font_color('red')
        formatRedLeft.set_align('left')
        formatLeft = workbook.add_format()
        formatLeft.set_align('left')
        formatLeftBold = workbook.add_format()
        formatLeftBold.set_bold()
        formatLeftBold.set_align('left')

        worksheet2.write(0, 0, 'Applying the model on some Benchmark Reviews:', formatLeftBold)
        worksheet2.write(2, 0, 'ReviewAbout', formatLeftBold)
        worksheet2.write(2, 1, 'ReviewScore', formatLeftBold)
        worksheet2.write(2, 2, 'Review', formatLeftBold)
        # print(benchmarkReviews)
        for s in range(noOfTopics):
            worksheet2.write(2, s + 3, 'Topic ' + str(s), formatLeftBold)
        worksheet2.write(2, noOfTopics + 3, 'Peak Topic', formatLeftBold)
        for s in range (len(benchmarkReviews)):
            worksheet2.write(s + 3, 0, str(benchmarkReviews[s][0]), formatLeft)
            worksheet2.write(s + 3, 1, str(benchmarkReviews[s][1]), formatLeft)
            worksheet2.write(s + 3, 2, str(benchmarkReviews[s][2]), formatLeft)
            for o in range(noOfTopics):
                worksheet2.write(s + 3, o + 3, benchmarkReviews[s][3][o], formatLeft)
            worksheet2.write(s+3, noOfTopics + 3, benchmarkReviews[s][3].index(max(benchmarkReviews[s][3])), formatLeft)
        workbook.close()

    reportIt()
