# QuoraPair
Identifying similar questions of Quora

Input: Two questions asked by users 
Ouput: Are they asking same questions or not?

E.g.
1.'What is procedure of conducting election?' is different question than 'What is procedure of conducting US election?' . However, they have high similarity.

2. 'What is your most loved place' is same questions as 'Which place do you love the most?' 

Classification Problem

Steps
1. cleaning irrelevant questions by checking question1 and question2 length
2. Basic features like length, Jackard similarity, Levenshtein distance, different ratios ,number of unique nouns etc.
3. Specific features like skewness and kurtosis, presence of specific words like :'again, $, differ, without, between, %'
4. Creating different models
5. Producing result on test data
