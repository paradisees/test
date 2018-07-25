import nltk

text="can not find reseller eaasily, distributor bogazici only listed products, no pictures. and also penta same. "
sens=nltk.sent_tokenize(text)
#print(sens)
words=[]
for sent in sens:
    words.append(nltk.word_tokenize(sent))
#print(words)
tags=[]
for tokens in words:
    tags.append(nltk.pos_tag(tokens))
#print(tags)
res=[]
for tag in tags:
    ners=nltk.ne_chunk(tag)
    res.append(ners)
print(res)
#print(res[0])
#print(res[1])
