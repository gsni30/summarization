from nltk.tokenize import sent_tokenize,word_tokenize
import sys
import re
from nltk.corpus import stopwords 
from string import punctuation
import math
import itertools
import pickle

def clean_text(article):
	for i in article:
		if ord(i)>128:
			article= article.replace(i,'')
	article= article.replace('\n',' ')
	article= article.replace('`','')
	article= article.replace('"','')
	article= article.replace("'",'')
	article= article.replace('.','')
	article= article.replace(',','')
	
	
	article= re.sub(r'[\w\.-]+@[\w\.-]+', '', article)# remove email addresses
	return article
	
	
def remove_stopwords(article):	
	sentences= sent_tokenize(article)
	word_sent = [word_tokenize(s) for s in sentences] 
	stop= set(stopwords.words('english')+list(punctuation))
	clean_sentence =''
	for sent in word_sent:#construct sentences from word list generated after removing stopwords
		se= ''
		for word in sent:
			if not word in stop:
				se= se+ word+' '
		clean_sentence+= se
	return clean_sentence
	
def n_grams_match(ref_text, sys_text, n):
	word_sys_text=word_tokenize(sys_text)
	num_sys_words= len(word_sys_text)
	ngrams_sys=[]
	ngrams_ref=[]
	i=0
	num_ref_ngrams=1.0
	num_sys_ngrams=num_sys_words-n+1
	matched_ngrams=1.0
	while i<num_sys_words-n+1:
		ngrams_sys.append(word_sys_text[i:i+n])
		i+=1
	
	for text in ref_text:
		word_ref_text= word_tokenize(text)
		num_ref_words= len(word_ref_text)
		ngrams_ref=[]
		j=0
		while j<num_ref_words-n+1:
			ngrams_ref.append(word_ref_text[j:j+n])		
			j+=1
		num_ref_ngrams+=(num_ref_words-n+1)
		
		
	for ngram in ngrams_sys:
		if ngram in ngrams_ref:
			matched_ngrams+=1
	# print matched_ngrams, num_sys_ngrams, num_ref_ngrams
	
	bp=0
	if num_sys_words<num_ref_words:
		bp= math.exp(1- num_ref_words/num_sys_words)
	else:
		bp=1.0
	
	
	# file.close()
	dic= {'bp':bp, 'num_matched':(matched_ngrams*1.0)/num_sys_ngrams, 'precision':(matched_ngrams*1.0)/num_ref_ngrams, 'recall':(matched_ngrams*1.0)/num_sys_ngrams}
	return dic
	
def lcs(line, text_file):
	X= word_tokenize(line)
	Y= word_tokenize(text_file)
	def LCS(X, Y):
		m = len(X)
		n = len(Y)
		C = [[0.0 for j in range(n+1)] for i in range(m+1)]
		for i in range(1, m+1):
			for j in range(1, n+1):
				if X[i-1] == Y[j-1]: 
					C[i][j] = C[i-1][j-1] + 1
				else:
					C[i][j] = max(C[i][j-1], C[i-1][j])
		return C

	def backTrack(C, X, Y, i, j):
		if i == 0 or j == 0:
			return ""
		elif X[i-1] == Y[j-1]:
			return backTrack(C, X, Y, i-1, j-1) + X[i-1]
		else:
			if C[i][j-1] > C[i-1][j]:
				return backTrack(C, X, Y, i, j-1)
			else:
				return backTrack(C, X, Y, i-1, j)

	
	max1=0		#max no. of words found
	k=0
	m= len(line)
	n= len(text_file)
	C = LCS(line, text_file)
	k=C[m][n]
	if max1 < k:
		max1 = k
			 
	return max1

def skip_bigrams_rouge(ref_text, sys_text):
	bigrams_ref=[]
	bigrams_sys=[]
	word_sys_text=word_tokenize(sys_text)
	num_sys_words= len(word_sys_text)
	bigrams_sys= itertools.combinations(word_sys_text, 2)
	bigrams_sys= list(bigrams_sys)
	
	word_ref_text=word_tokenize(ref_text)
	num_ref_words= len(word_ref_text)
	bigrams_ref= itertools.combinations(word_ref_text, 2)
	bigrams_ref= list(bigrams_ref)
	
	num_matched_bigrams=0.0
	matched_bigrams= list(set(bigrams_ref).intersection(bigrams_sys))
	num_matched_bigrams=len(matched_bigrams)
	
	rskip2= num_matched_bigrams*1.0/len(bigrams_ref)
	pskip2= num_matched_bigrams*1.0/len(bigrams_sys)
	
	fskip2= 2*rskip2*pskip2/(rskip2+pskip2)
	
	return fskip2
	
	
def ngrams_rouge(ref_text, system_text, N):
	dict= {}
	sum=0.0
	for n in range(1, N+1):
		dic= n_grams_match(ref_text,system_text,n)
		bp = dic['bp']
		p= dic['num_matched']
		sum+=(math.log(p))
		dict['bleu'+str(n)] =sum/n
		dict['precision'+str(n)] =dic['precision']
		dict['recall'+str(n)] =dic['recall']
	return dict

def lcs_sent_level_rouge(reference_text, system_text):
	ref_sent= sent_tokenize(reference_text)
	sys_sent= sent_tokenize(system_text)
	
	lcs_score=0.0
	
	for ref in ref_sent:
		for sys in sys_sent:
			lcs_score += lcs(sys, ref)
	r_lcs= lcs_score/len(system_text)
	p_lcs= lcs_score/len(reference_text)
	
	beta= p_lcs/r_lcs
	
	f_lcs= (1+beta*beta)*r_lcs*p_lcs/(r_lcs + beta*beta*p_lcs)
	return f_lcs

if __name__ == '__main__':	
	ref_text=[]
	N=3
	sources = pickle.load( open( "ref_summaries_dir_struct.pkl", "rb" ) )
	scores=[]
	for src in sources:
		source= sources[src]
		for date in source:
			try:
				reference_file= open('ref_summaries/'+src+'.'+date+'.doc2vec_hier.txt', 'r')
				system_file= open('sys_summaries/doc2vec_hier/'+src+'.'+date+'.doc2vec_hier.txt', 'r')
				reference_text= reference_file.read()
				system_text= system_file.read()
					
				reference_text= clean_text(reference_text)
				system_text= clean_text(system_text)
				
				skip_bigrams_stats= skip_bigrams_rouge(reference_text, system_text)
				lcs_score= lcs_sent_level_rouge(reference_text, system_text)
				ngrams_score= ngrams_rouge(reference_text, system_text,N)
				score={}
				score["file"]= src+'.'+date+'.doc2vec_hier.txt'
				score["skip_bigrams_stats"]= skip_bigrams_stats
				score["lcs_score"]=lcs_score
				score["unigram_precision"]= ngrams_score["precision1"]
				score["bigram_precision"]= ngrams_score["precision2"]
				score["trigram_precision"]= ngrams_score["precision3"]
				score["unigram_recall"]= ngrams_score["recall1"]
				score["bigram_recall"]= ngrams_score["recall2"]
				score["trigram_recall"]= ngrams_score["recall3"]
				score["unigram"]= ngrams_score["bleu1"]
				score["bigram"]= ngrams_score["bleu2"]
				score["trigram"]= ngrams_score["bleu3"]
				scores.append(score)
				print scores
				reference_file.close()
				system_file.close()
			except:
				pass	
	pickle.dump(scores, open( "rouge_scores_doc2vec_hierarchical.pkl", "wb" ))