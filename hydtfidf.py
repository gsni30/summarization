#------------------hybrid tf-idf---------------

import MySQLdb
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import string ,re
import math
import operator
import cosim

_stopwords = stopwords.words('english')
_stopwords.extend(string.punctuation)
_stopwords.append("rt") 


#----------Term Frequency ----------
def _compute_frequencies(word_sent):
		""" 
		Compute the frequency of each of word.
		Input: word_sent, a list of sentences already tokenized.
		Output: freq, a dictionary where freq[w] is the frequency of w.
		"""
		freq = defaultdict(int)
		for s in word_sent:
			for word in s:
				if word not in _stopwords:
					freq[word] += 1
				# frequencies normalization and fitering
		return freq


def term_freq(word_dict , no_of_word):
	tf = defaultdict(float)
	for item in word_dict:
		tf[item[0]]= float(item[1])/float(no_of_word)
	return tf

def inverse_freq(tweetset, word_dict,tweetno):
	#cursor.execute("select count(*) from %s;" % (tablename)) # ctweet has punctuations,stopwords,hyperlinks removed
	#db.commit()
	#tweetno=cursor.fetchall()
	totdoc= tweetno # total #docs=tot #tweets
	#print totdoc[0]
	idf = defaultdict(float)
	for word in word_dict :
		idf[word[0]]=0
		#print word

	#print word_dict[0][0]
	for word in word_dict :
		for tweet in tweetset:
			#print word[0]+ " : "+tweet[1]
			if word[0] in tweet[1]:
				idf[word[0]]=idf[word[0]]+1

	for key, value in idf.iteritems():
		'''if value==0:
			print key
		else:'''
		val =  math.log(1 + totdoc/value)
		idf[key]=val
		
	return idf

#-----------------------------------MAIN -------------------------------------------------

#------------------Making document of all tweets to calcuate tf ----------------------

def begin(hashtag):
	db= MySQLdb.connect("localhost","root","shruti","osm")
	cursor = db.cursor()
	tablename = hashtag
	print hashtag
	cursor.execute("select sno,ctweet from %s;" % (tablename)) # ctweet has punctuations,stopwords,hyperlinks removed
	db.commit()
	tweetset=cursor.fetchall()
	tweetno=len(tweetset)
	tweettext=''
	for tweet in tweetset:
		#print tweet[1]
		tweetq = re.sub(' +',' ',tweet[1])
		#print tweetq
		tweettext = tweetq+' . '+tweettext
	#print tweettext

	sents = sent_tokenize(tweettext)

	word_sent = [word_tokenize(s) for s in sents]
	#print word_sent

	_freq = _compute_frequencies(word_sent)	
	#print _freq
	#-----------sorting the words acc to frequency ---------------
	sorted_x = sorted(_freq.items(), key=operator.itemgetter(1))

	'''for key, value in sorted_x.iteritems():
		print key+":"+str(value)
	'''
	# --- calculating #total words -----
	total_words=0
	for item in sorted_x:
		#print item[0]+':'+str(item[1])
		total_words=total_words+item[1]

	# --- calculating #tf of all words -----
	_tf = term_freq(sorted_x , total_words)
	'''for key, value in _tf.iteritems():
		print key+":"+str(value)
	print "\n\n\n"'''

	# --- calculating #idf of all words -----
	_idf = inverse_freq(tweetset, sorted_x,tweetno)
	'''for key, value in _idf.iteritems():
		print str(key)+":"+str(value)
	print "\n\n\n"'''

	# --- calculating #tfidf of all words -----
	_tfidf = defaultdict(float)
	for key,value in sorted_x:	
		_tfidf[key]=_tf[key]*_idf[key]
		#print key,_tfidf[key],_tf[key],_idf[key]

	# --- ranking of all tweets on tfidf score -----
	update_query= "UPDATE "+tablename+" SET hyd_tfidf= CASE sno "
	sno_array= "("
	for tweet in tweetset:
		summ=0
		#print tweet[1]
		words= tweet[1].split()
		#print words
		for word in words:
			#print word
			summ=summ+_tfidf[word]
			#print word,_tfidf[word]
		#print tweet[1], sum
		update_query+= " WHEN "+str(tweet[0])+" THEN "+str(round(summ,8))
		sno_array+= str(tweet[0])
		sno_array+= ","
		'''try:
			cursor.execute("update %s set hyd_tfidf=%s where sno=%s" % (tablename, `round(summ,10)`,tweet[0]))
			db.commit()
		except:
			print "error"
		'''
	sno_array= sno_array[:-1]

	sno_array+=")"

	update_query+= " END WHERE sno in "+sno_array
	cursor.execute(update_query)
	db.commit()
	cursor.execute("select distinct tweet, hyd_tfidf from %s order by hyd_tfidf desc limit 25;" % (tablename))	
	db.commit()
	toptweets=cursor.fetchall();
	i=1
	bool_arr=[0] * 25
	'''for toptweet in toptweets:
		print i,toptweet[0],toptweet[1]
		i=i+1
	'''
	i=1
	tweetct=25
	output={}
	for x in range(tweetct):
		if bool_arr[x] == 0:
			bool_arr[x] = 1;
			output[i]=toptweets[x][0]

			i=i+1
			if i==21:
				break
			for z in range(x+1, tweetct):
				if cosim.cosine_similarity(toptweets[x][0],toptweets[z][0]) > 0.8 :
					bool_arr[z]=1	
	print output
	return output


	#-----------check which log -------------
	#print math.log(2,2)
	 
