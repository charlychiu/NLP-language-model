import nltk

from nltk.corpus import brown

# an nltk.FreqDist() is like a dictionary,
# but it is ordered by frequency.
# Also, nltk automatically fills the dictionary
# with counts when given a list of words.

freq_brown = nltk.FreqDist(brown.words())

list(freq_brown.keys())[:20]
freq_brown.most_common(20)

# an nltk.ConditionalFreqDist() counts frequencies of pairs.
# When given a list of bigrams, it maps each first word of a bigram
# to a FreqDist over the second words of the bigram.

cfreq_brown_2gram = nltk.ConditionalFreqDist(nltk.bigrams(brown.words()))

# conditions() in a ConditionalFreqDist are like keys()
# in a dictionary

cfreq_brown_2gram.conditions()

# the cfreq_brown_2gram entry for "my" is a FreqDist.

cfreq_brown_2gram["my"]

# here are the words that can follow after "my".
# We first access the FreqDist associated with "my",
# then the keys in that FreqDist

cfreq_brown_2gram["my"].keys()

# here are the 20 most frequent words to come after "my", with their frequencies

cfreq_brown_2gram["my"].most_common(20)

# an nltk.ConditionalProbDist() maps pairs to probabilities.
# One way in which we can do this is by using Maximum Likelihood Estimation (MLE)

cprob_brown_2gram = nltk.ConditionalProbDist(cfreq_brown_2gram, nltk.MLEProbDist)

# This again has conditions() wihch are like dictionary keys

cprob_brown_2gram.conditions()

# Here is what we find for "my": a Maximum Likelihood Estimation-based probability distribution,
# as a MLEProbDist object.

cprob_brown_2gram["my"]

# We can find the words that can come after "my" by using the function samples()

cprob_brown_2gram["my"].samples()

# Here is the probability of a particular pair:



cprob_brown_2gram["my"].prob("own")

#####

# We can also compute unigram probabilities (probabilities of individual words)

freq_brown_1gram = nltk.FreqDist(brown.words())

len_brown = len(brown.words())


def unigram_prob(word):

return freq_brown_1gram[ word] / len_brown



#############

# The contents of cprob_brown_2gram, all these probabilities, now form a

# trained bigram language model. The typical use for a language model is

# to ask it for the probabillity of a word sequence

# P(how do you do) = P(how) * P(do|how) * P(you|do) * P(do | you)

prob_sentence = unigram_prob("how") * cprob_brown_2gram["how"].prob("do") * cprob_brown_2gram["do"].prob("you") * \ cprob_brown_2gram["you"].prob("do")

print(prob_sentence)
# result: 1.5639033871961e-09


###############

# We can also use a language model in another way:

# We can let it generate text at random

# This is not so useful, but can be insightful into what it is that

# the language model has been learning



cprob_brown_2gram["my"].generate()

# We can use this to generate text at random
# based on a given text of bigrams.
# Let's do this for the Sam "corpus"

corpus = """<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>"""

words = corpus.split()
cfreq_sam = nltk.ConditionalFreqDist(nltk.bigrams(words))
cprob_sam = nltk.ConditionalProbDist(cfreq_sam, nltk.MLEProbDist)

word = "<s>"
for index in range(50):
word = cprob_sam[ word].generate()
print(word, end = " ")

print("\n")

# Not a lot of variety. We need a bigger corpus.
# What kind of genres do we have in the Brown corpus?
brown.categories()

# Let's try Science Fiction.
cfreq_scifi = nltk.ConditionalFreqDist(nltk.bigrams(brown.words(categories = "science_fiction")))
cprob_scifi = nltk.ConditionalProbDist(cfreq_scifi, nltk.MLEProbDist)

word = "in"
for index in range(50):
word = cprob_scifi[ word ].generate()
print(word, end = " ")
print

# try this with other Brown corpus categories.


# Here is how to do this with NLTK books:
import nltk
from nltk.book import *

def generate_text(text, initialword, numwords):
bigrams = list(nltk.ngrams(text, 2))
cpd = nltk.ConditionalProbDist(nltk.ConditionalFreqDist(bigrams), nltk.MLEProbDist)


word = initialword
for i in range(numwords):
print(word, end = " ")
word = cpd[ word].generate()

print(word)

# Holy Grail
generate_text(text6, "I", 100)
# sense and sensibility
generate_text(text2, "I", 100)