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

    topics = [15,16,17,18,19,20,21,22,23,24,25]
    pickle.dump(topics, open('LDA_03_Training_noOfTopics.list', 'wb'))
    now = datetime.datetime.now()
    currDateTime = (now.strftime("%d%m%Y_%H%M%S"))
    dir = 'LDA_Runs/01_All/' + currDateTime
    pickle.dump(dir, open('LDA_03_Training_LastRunTime.string', 'wb'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    for tps in range(len(topics)):
        setNoOfTopics = topics[tps]
        setAlpha = 0.1
        setEta = 0.1
        setFitBundle = False

        setTempRun = '_engsample' #change benchmark review size also

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

        dct = corpora.dictionary.Dictionary.load('LDA_02_Preprocessing_Dictionary.dictionary')
        corpus = pickle.load(open('LDA_02_Preprocessing_Corpus.corpus', 'rb'))
        keywordsConstructAll = pickle.load(open('LDA_02_Preprocessing_keywordsConstructAllNew.list', 'rb'))
        keywordsConstructAllIDsInDct = pickle.load(open('LDA_02_Preprocessing_keywordsConstructAllIDsInDctNew.list', 'rb'))
        listNoun = pickle.load(open('LDA_02_Preprocessing_listNounNew.list', 'rb'))
        listAdj = pickle.load(open('LDA_02_Preprocessing_listAdjNew.list', 'rb'))
        listVerb = pickle.load(open('LDA_02_Preprocessing_listVerbNew.list', 'rb'))
        listNounIds = pickle.load(open('LDA_02_Preprocessing_listNounIdsNew.list', 'rb'))
        listAdjIds = pickle.load(open('LDA_02_Preprocessing_listAdjIdsNew.list', 'rb'))
        listVerbIds = pickle.load(open('LDA_02_Preprocessing_listVerbIdsNew.list', 'rb'))
        dctOpsLog = pickle.load(open('LDA_02_Preprocessing_dctOpsLog.list', 'rb'))

        probSkew=15

        # for t in range(noOfTopics):
        #     wordProbs_temp = []
        #     for d in range(len(dct)):
        #         wordProbs_temp.append(10)
        #         if t == 0:
        #             if dct[d] in keywordsConstruct1:
        #                 wordProbs_temp[d] = probSkew
        #         if t == 1:
        #             if dct[d] in keywordsConstruct2:
        #                 wordProbs_temp[d] = probSkew
        #         if t == 2:
        #             if dct[d] in keywordsConstruct3:
        #                 wordProbs_temp[d] = probSkew
        #         if t == 3:
        #             if dct[d] in keywordsConstruct4:
        #                 wordProbs_temp[d] = probSkew
        #         if t == 4:
        #             if dct[d] in keywordsConstruct5:
        #                 wordProbs_temp[d] = probSkew
        #         if t == 5:
        #             if dct[d] in keywordsConstruct6:
        #                 wordProbs_temp[d] = probSkew
        #     wordProbs.append(wordProbs_temp)

        wordProbs = []
        for d in range(len(dct)):
            wordProbs.append(10)
            if dct[d] in keywordsConstructAll:
                wordProbs[d] = probSkew

        # Step 4: Train the LDA model
        lda_model = LdaModel(corpus=corpus,
                                 id2word=dct,
                                 random_state=100,
                                 num_topics=setNoOfTopics,
                                 passes=10, #2,
                                 chunksize=1000,
                                 # batch=False,
                                 alpha = setAlpha, #alpha='asymmetric', # alpha=1/2, #alpha=[0.5,0.5], #greater than 1 gives docs all topics alomost equal prob
                                 decay=0.5,
                                 offset=1.5, #64,
                                 eta = setEta, #wordProbs, #eta=None, # eta=1/370,
                                 eval_every=0,
                                 iterations=100,
                                 gamma_threshold=0.001,
                                 per_word_topics=True)

        # save the model
        lda_model.save('LDA_03_Training_Model.model')

        # See the topics
        lda_model.print_topics(-1)
        # print(lda_model.get_topic_terms(0,370))
        # print(lda_model.get_topic_terms(1,370))
        # print(lda_model.get_term_topics(102,1))

        def reportIt(trainSetProps='', alphaProps='', etaProps='', trunTerms=len(dct)):
            # newpath = r'C:\Program Files\arbitrary'
            # if not os.path.exists(newpath):
            #     os.makedirs(newpath)
            if tps == 0:
                shutil.copy2('LDA_00_CorpusAnalysis.py', dir + '/LDA_00_CorpusAnalysis_' + currDateTime + '.py')
                shutil.copy2('LDA_00_CorpusAnalysis_dctMaster.csv', dir + '/LDA_00_CorpusAnalysis_dctMaster_' + currDateTime + '.csv')
                shutil.copy2('LDA_01_ReviewsPicker.py', dir + '/LDA_01_ReviewsPicker_' + currDateTime + '.py')
                shutil.copy2('LDA_01_ReviewsPicker_Master_Data_for_training.csv', dir + '/LDA_01_ReviewsPicker_Master_Data_for_training_' + currDateTime + '.csv')
                shutil.copy2('LDA_02_Preprocessing.py', dir + '/LDA_02_Preprocessing_' + currDateTime + '.py')
                shutil.copy2('LDA_02_Preprocessing_Corpus.corpus', dir + '/LDA_02_Preprocessing_Corpus_' + currDateTime + '.corpus')
                shutil.copy2('LDA_02_Preprocessing_Dictionary.dictionary', dir + '/LDA_02_Preprocessing_Dictionary_' + currDateTime + '.dictionary')
            dir2 = 'LDA_Runs/01_All/' + currDateTime + '/noOfTopics_' + str(setNoOfTopics)
            if not os.path.exists(dir2):
                os.makedirs(dir2)
            currDateTime2 = currDateTime + '_noOfTopics_' + str(setNoOfTopics)
            shutil.copy2('LDA_03_Training.py', dir2 + '/LDA_03_Training_' + currDateTime2 + '.py')
            shutil.copy2('LDA_03_Training_Model.model', dir2 + '/LDA_03_Training_Model_' + currDateTime2 + '.model')
            shutil.copy2('LDA_03_Training_Model.model.expElogbeta.npy', dir2 + '/LDA_03_Training_Model.model.expElogbeta_' + currDateTime2 + '.npy')
            shutil.copy2('LDA_03_Training_Model.model.id2word', dir2 + '/LDA_03_Training_Model.model_' + currDateTime2 + '.id2word')
            shutil.copy2('LDA_03_Training_Model.model.state', dir2 + '/LDA_03_Training_Model.model_' + currDateTime2 + '.state')
            workbook = xlsxwriter.Workbook(dir2 + '/LDA_03_Training_Report_' + currDateTime2 + '.xlsx')
            worksheet = workbook.add_worksheet()

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
            formatYellowLeft = workbook.add_format()
            formatYellowLeft.set_bg_color('yellow')
            formatYellowLeft.set_align('left')

            worksheet.write(0, 0, 'Training Set Properties:', formatLeftBold)
            worksheet.write(1, 0, trainSetProps, formatLeft)
            worksheet.write(2, 0, 'Alpha (Reviews-Topics Probability Distribution Prior):', formatLeftBold)
            worksheet.write(3, 0, alphaProps, formatLeft)
            worksheet.write(4, 0, 'Eta (Terms-Topics Probability Distribution Prior):', formatLeftBold)
            worksheet.write(5, 0, etaProps, formatLeft)
            worksheet.write(6, 0, 'No. of Topics:', formatLeftBold)
            worksheet.write(7, 0, setNoOfTopics, formatLeft)
            worksheet.write(8, 0, 'Corpus length:', formatLeftBold)
            worksheet.write(9, 0, len(corpus), formatLeft)
            worksheet.write(10, 0, 'Dictionary Operations:', formatLeftBold)
            dctOps = ''
            for ops in range(len(dctOpsLog)):
                dctOps = dctOps + str(dctOpsLog[ops]) + ' /***/ '
            worksheet.write(11, 0, dctOps)
            worksheet.write(12, 0, 'Topics Terms Distribution:', formatLeftBold)
            if trunTerms == len(dct):
                worksheet.write(13, 0, 'It is not truncated. All topics have all words of dictionary', formatLeft)
            else:
                worksheet.write(13, 0, 'It is truncated for top ' + str(trunTerms) + ' terms in each topic.', formatLeft)

            for s in range(setNoOfTopics):
                worksheet.write(15, s * 3, 'Topic ' + str(s), formatLeftBold)
                worksheet.write(16, s * 3, 'Word', formatLeftBold)
                worksheet.write(16, (s * 3) + 1, 'Probability', formatLeftBold)
                vec = lda_model.get_topic_terms(s, trunTerms)
                for h in range(trunTerms):
                    if dct.id2token[vec[h][0]] in keywordsConstructAll:
                        worksheet.write(h + 17, (s * 3), dct.id2token[vec[h][0]], formatRedLeft)
                    else:
                        worksheet.write(h + 17, (s * 3), dct.id2token[vec[h][0]], formatLeft)
                    if vec[h][1] * 100 >= 5:
                        worksheet.write(h + 17, (s * 3) + 1, vec[h][1], formatYellowLeft)
                    else:
                        worksheet.write(h + 17, (s * 3) + 1, vec[h][1], formatLeft)

            workbook.close()

        reportIt('Reviews are bundled for user and only those are considered for training which possess some word in dictionary.',
                 'Alpha is 0.01',
                 'Eta is 0.01',
                 len(dct))

# takes max 20 mins