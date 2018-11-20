import nltk
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

# with open('sentence_list.pickle', 'rb') as file:
#     sentence_list = pickle.load(file)
#
# cfreq_brown_2gram = nltk.ConditionalFreqDist(nltk.bigrams(sentence_list))
# cfreq_brown_2gram.conditions()
#
# file = open('cfreq_brown_2gram.pickle', 'wb')
# pickle.dump(cfreq_brown_2gram, file)
# file.close()


# with open('cfreq_brown_2gram.pickle', 'rb') as file:
#     cfreq_brown_2gram = pickle.load(file)


# cprob_brown_2gram = nltk.ConditionalProbDist(cfreq_brown_2gram, nltk.MLEProbDist)
# cprob_brown_2gram.conditions()
# file = open('cprob_brown_2gram.pickle', 'wb')
# pickle.dump(cprob_brown_2gram, file)
# file.close()

with open('cprob_brown_2gram.pickle', 'rb') as file:
    cprob_brown_2gram = pickle.load(file)

# print(cprob_brown_2gram["我"].prob("們"))

# freq_brown_1gram = nltk.FreqDist(sentence_list)
#
# file = open('freq_brown_1gram.pickle', 'wb')
# pickle.dump(freq_brown_1gram, file)
# file.close()
#
with open('freq_brown_1gram.pickle', 'rb') as file:
    freq_brown_1gram = pickle.load(file)

# len_brown = len(sentence_list)
#
# file = open('len_brown.pickle', 'wb')
# pickle.dump(len_brown, file)
# file.close()

with open('len_brown.pickle', 'rb') as file:
    len_brown = pickle.load(file)


def unigram_prob(word):
    return freq_brown_1gram[word] / len_brown


testing_data = ['川普任職於白宮', '一種基本人權', '人人有保持意見不受干預之權利', '我思故我在', '柏拉圖式戀愛', '奇美博物館主場館', '供應鏈上製造了混亂', '加州史丹佛大學理學博士生',
                '沒想到開季又是內角觸身', '雲端基礎設施服務市場']
for data in testing_data:
    prob_sentence1 = unigram_prob(data[0])
    for i in range(1, len(data)):
        prob_sentence1 *= cprob_brown_2gram[data[i - 1]].prob(data[i])
    print(prob_sentence1)

# prob_sentence = unigram_prob("民") * cprob_brown_2gram["民"].prob("主") * cprob_brown_2gram["主"].prob("國")* cprob_brown_2gram["國"].prob("家")
# print(prob_sentence) # 0.28532904004829945
