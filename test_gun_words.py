data = [line.strip().split('\t') for line in open('articles').readlines()]
stopwords = open('stopwords.txt','r').readlines()
stripped_stopwords = []
for word in stopwords :
	stripped_stopwords.append(word.strip())

gun_words = []
gun_words_freq = []
non_gun_words = []
non_gun_words_freq = []
#test

key_words = []
for (label, article) in data :
	if label == "1" :
		words = article.split()
		for word in words :
			word = word.lower()
			if word in stripped_stopwords :
				#do nothing
				x=1
			elif word in gun_words :
				gun_words_freq[gun_words.index(word)] = gun_words_freq[gun_words.index(word)] + 1
				if gun_words_freq[gun_words.index(word)] > 1000 and not (word in key_words) :
					key_words.append(word)
					print "New word: " + word
			else :
				gun_words.append(word)
				gun_words_freq.append(1)

"""
Order of words reaching the 1000 threshold for gun-related articles (label =="1"):
police
said
said.
shot
man
shooting
according
gun
block
p.m.
two
county
one
found
home
told
taken
victim
officers
hospital
say
woman
people
around

Order of words reaching the 1000 threshold for non-gun-related articles (label =="0"):
-
news
...
2014
ad
id:
new
home
sep
said
day
one
said.
--
police
-h-
service
county
man
two
also
would
"""