def textParse(bigString):  # input is big string, #output is word list
    import re
    from nltk.corpus import stopwords
    listOfTokens = re.split(r'\W*', bigString)
    text= [tok.lower() for tok in listOfTokens if len(tok) > 2]
    #print(text)
    filtered = ''
    res=[]
    for w in text:
        # print(w)
        if w not in stopwords.words('english'):
            res.append(filtered.join(w))
    return res


print(textParse('I like to use lenovo device'))