import nltk
from nltk.corpus import stopwords
import pickle

# fileSet = ['zh_wiki_00', 'zh_wiki_01']
fileSet = ['zh_wiki_01']
word_list = []
sentence_list = []

# for file in fileSet:
#     with open(file, 'r', encoding='utf-8') as fs:
#         line = fs.readline()
#         while line:
#             # print(line.strip()))
#             sentence_list.extend(line.strip())
#             line = fs.readline()
#
# file = open('sentence_list.pickle', 'wb')
# pickle.dump(sentence_list, file)
# file.close()

with open('sentence_list.pickle', 'rb') as file:
    sentence_list = pickle.load(file)

trigram_result = nltk.trigrams(sentence_list)
condition_paris = (((w0, w1), w2) for w0, w1, w2 in trigram_result)
cfreq_brown_3gram = nltk.ConditionalFreqDist(condition_paris)

file = open('cfreq_brown_3gram.pickle', 'wb')
pickle.dump(cfreq_brown_3gram, file)
file.close()

# with open('cfreq_brown_3gram.pickle', 'rb') as file:
#     cfreq_brown_3gram = pickle.load(file)

cfreq_brown_3gram.conditions()
cprob_brown_3gram = nltk.ConditionalProbDist(cfreq_brown_3gram, nltk.MLEProbDist)
# file = open('cprob_brown_3gram.pickle', 'wb')
# pickle.dump(cprob_brown_3gram, file)
# file.close()

with open('cprob_brown_3gram.pickle', 'rb') as file:
    cprob_brown_3gram = pickle.load(file)

cprob_brown_3gram.conditions()

print(cprob_brown_3gram["歐"]["巴"].prob("馬"))
print(cprob_brown_3gram["歐"]["吉"].prob("桑"))

freq_brown_1gram = nltk.FreqDist(sentence_list)
len_brown = len(sentence_list)


def unigram_prob(word):
    return freq_brown_1gram[word] / len_brown

# prob_sentence = unigram_prob("小") * cprob_brown_3gram["小"].prob("琉") * cprob_brown_3gram["琉"].prob("球")
#
# print(prob_sentence) # 0.28532904004829945
# bigram_finder = nltk.BigramCollocationFinder.from_words(fs)
# bigram_finder = bigram_finder.nbest(score_fn=nltk.BigramAssocMeasures.chi_sq, n=1000)
# print(bigram_finder)
# test = nltk.bigrams(word_list)
# for i in test:
#     print(i)
