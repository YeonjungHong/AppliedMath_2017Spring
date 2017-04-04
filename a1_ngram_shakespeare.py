# 2017-Mar-25
# Applied math assignment 1
# Yeonjung Hong

import re
import matplotlib.pyplot as plt
import numpy as np


#1. File reading
with open('shakespeare.txt', 'r') as f: # using "with", the file will be automatically closed after.
    lines = f.readlines()

#2. The 1st clean-up: lower case
lines_1 = [ x.lower() for x in lines]

#3. The 2nd clean-up: remove non alphabets
lines_2 = [ re.sub(r'\W+', ' ', x) for x in lines_1 ]
# lines_2 = [ re.sub(r'[^A-Za-z]+', ' ', x) for x in lines_1 ]

#4. Split the string along the spaces into a list of sub-strings -> create a dictionary
lines_split = [ x.split(' ') for x in lines_2] # split delimited by a whitespace
flatten = lambda l: [item for sublist in l for item in sublist] # create a lambda function to flatten ndarray
word_seq = flatten(lines_split) # flatten the sentences into a word list
word_seq = filter(None, word_seq) # delete empty strings

#5. Unigrams
unigram, uni_counts = np.unique(word_seq, return_counts=True) # get unigram list and its corresponding counts
sorted_uni = sorted(zip(uni_counts, unigram), reverse=True) # sort unigram list in a descending order

#6. Make a bar plot
top30_uni_count = [ int(pair[0]) for pair in sorted_uni[:30]] # top 30 unigram counts
top30_uni_word = [ pair[1] for pair in sorted_uni[:30]] # top 30 unigram list
pos = np.arange(30)

plt.bar(pos, top30_uni_count, alpha=0.5)
plt.xticks(pos, top30_uni_word)
plt.ylabel('Frequency')
plt.title('Top 30 unigrams in Shakespeare.txt')

plt.show()

#7. Bigrams

# create a bigram pair
pair_list = zip(word_seq[:-1], word_seq[1:]) # create a word pair list with word sequences
bigram = list(set(pair_list)) # unique list of bigram

# create an empty table for bigram count
bi_count = np.zeros((len(unigram), len(unigram)))
# More efficient way to do is to make a sparse matrix where only nonzero elements are used for matrix creation 
# and other indices are implicitly zero.

for pair in pair_list:
    left_word = pair[0]; right_word = pair[1] # left and right word in a word pair
    left_idx = np.where(unigram == left_word) # get indices for both left/right words in a unigram list
    right_idx = np.where(unigram == right_word)
    bi_count[left_idx, right_idx] += 1 # add up the counts for bigram count matrix <= this takes A LOT OF TIME

# extract top 30 pairs
top30_bi_idx = np.unravel_index(bi_count.flatten().argsort()[-30:][::-1], np.shape(bi_count))
top30_bi_count = bi_count[top30_bi_idx]
top30_bi_word = zip(unigram[top30_bi_idx[0]], unigram[top30_bi_idx[1]])


plt.bar(pos, top30_bi_count, alpha=0.5)
plt.xticks(pos, top30_bi_word)
plt.ylabel('Frequency')
plt.title('Top 30 bigrams in Shakespeare.txt')

plt.show()

#8. Create a most likely "sentence" that is 30 words in total starting with a given word and print it out.
def gen_sent_bicount_matrix(init_word):
    len_sent = 30
    word = init_word.lower()
    sent = [word]
    while len(sent) < len_sent:
        idx = np.where(unigram == word)
        if all(idx):
            next_word = unigram[np.argmax(bi_count[idx,:])]
            sent.append(next_word)
            word = next_word
        else:
            sent = ['SORRY: This word does not exist in Shakespeare.txt']
            break
    full_sent = ' '.join(sent)
    return(full_sent)

gen_sent_bicount_matrix('i')
gen_sent_bicount_matrix('joke')

#8-1. Another method using DataFrame
import pandas as pd
pair_list_concat = [' '.join(x) for x in pair_list]
bigram_concat, bi_counts_concat = np.unique(pair_list_concat, return_counts=True)
left_list = [ x.split(' ')[0] for x in bigram_concat]
right_list = [ x.split(' ')[1] for x in bigram_concat]
d = {'left': left_list, 'right': right_list, 'count': bi_counts_concat}
df = pd.DataFrame(d)

def gen_sent_bicount_alt(init_word):
    len_sent = 30
    word = init_word.lower()
    sent = [word]
    while len(sent) < len_sent:
        candidate = df[df.left == word]
        idx_max = candidate['count'].idxmax()
        next_word = candidate['right'][idx_max]
        sent.append(next_word)
        word = next_word

    full_sent = ' '.join(sent)
    return(full_sent)

gen_sent_bicount_alt('yes')
gen_sent_bicount_alt('about')

# The problem for this approach is
# 1) Too many stop words are included? (not a critical point)
# 2) common words have a tendency to be followed by another common word when it comes to bigram counting.
#   => common bigram loop
# 3) Sentence begin & end are not considered. N-gram counting should be carried out within a sentence.

