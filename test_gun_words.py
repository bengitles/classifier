data = [line.strip().split('\t') for line in open('articles').readlines()]

gun_words = []
gun_words_freq = []
non_gun_words = []
non_gun_words_freq = []
#test
for (label, article) in data :
	if label == "1" :
		words = article.split()
		for word in words :
			if word in gun_words :
				gun_words_freq[gun_words.index(word)] = gun_words_freq[gun_words.index(word)] + 1
			else :
				gun_words.append(word)
				gun_words_freq.append(1)
				print "test"

maxFreq = 0
for i in range (0,20) :
	for freq in gun_words_freq :
		print freq
		if freq >= maxFreq :
			print freq
			maxFreq = freq
	print gun_words[gun_words_freq.index(maxFreq)] + ": " + str(maxFreq)
	gun_words_freq[gun_words_freq.index(maxFreq)] = 0
	maxFreq = 0