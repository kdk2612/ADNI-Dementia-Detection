# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:04:22 2016

@author: kkothari
"""

import nltk
from multiprocessing import Pool
from itertools import chain
import string
import re
import os
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from itertools import groupby
from nltk import pos_tag, word_tokenize
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import unicodedata
csv.field_size_limit(1000000000)
#########################################SET Directory############################

current_dir = os.getcwd()
check_dir = os.path.basename(os.path.normpath(current_dir))
if check_dir == 'capstone_karan':
    path = os.getcwd()
else:
    os.chdir(r'capstone_karan')
    path = os.getcwd()


################################################################################
# Wordlists for replacement

def read_wordlist(wl):
    #return dict( [ l.strip().split(',')
    return [ tuple(l.strip().split(','))
             for l in open('%s/%s'%(r'wordlist',wl), 'r',encoding='utf-8') ] # change the path


#WL_w_ngrams     = read_wordlist('w_ngrams.txt')
#WL_stop           = read_wordlist('stop.txt')
#WL_days     = read_wordlist('days.txt')
#WL_lemma_ignore   = read_wordlist('lemmatize_ignore.txt')
#WL_states = read_wordlist('states.txt')
#WL_abbr_states = read_wordlist('state_abbrv.txt')
#WL_month_abbrv = read_wordlist('month_abbrv.txt')


################################################################################
# Generalized functions used by specific ones below

def replace_words(string, wordlist, delim=' '):
    """Replace single words OR multi-word expressions. Either will work
    with this method.

    Each word in the given wordlist is padded by the given delim, then replaced
    throughout the string. It might seem simpler to split the string by the
    delimiter, but doing it this way allows us to do multi-word expressions
    without a separate function.

    Args:
        string: the string (document)
        wordlist: dict of words or multi-word expression strings as keys,
                  replacements as values
        delim: (optional) delimiter of words in this context, default space ' '
    Returns:
        The new string with all replacements
    """
    # Add delims b/c first/last words
    new_string = delim + string + delim
    for word, repl in wordlist:#.items():
        word=word.lower()
        repl = repl.lower()
        word = delim + word + delim
        repl = delim if repl == '' else delim + repl + delim
        # Yes, I know the '1' limits it to 1, but sequential words weren't
        # matching right and this is the easiest way to fix it
        while word in new_string:
            new_string = new_string.replace( word, repl, 1 )
    return new_string[1:-1] # b/c the extra delims
    #words = string.split(delim)
    #for i in range(len(words)):
    #    if words[i] in wl:
    #        words[i] = wl[words[i]]
    #return delim.join([ w for w in words if w != '' ])
##################################LOAD###########################################

#from nltk.corpus import stopwords
#stop = stopwords.words('english')
##custom_stop = open(r'wordlist/stop.txt', 'r',encoding='utf-8')
##custom_stopl = custom_stop.read().splitlines()
#
##least_common= open(r'wordlist/least_common.txt', 'r',encoding='utf-8')
##least_commonl = least_common.read().splitlines()
#extra = [u'iv',u'i',u'ii',u'iii',u'v',u'vi',u'vii',u'viii',u'xi',u'ix',u'x',u'xiii',u'xviii']
#stop.extend(extra)
##stop.extend(custom_stopl)
##stop.extend(least_commonl)
#stop = [w.lower().strip() for w in stop]
#stop.append('whats')     
#stop = list(set(stop))   
#stop.sort()
#
#
#from nltk.corpus import gazetteers
#location = gazetteers.words()
##ner_stop = open(r'wordlist/ner_tag.txt','r',encoding='utf-8')
##ner_list = ner_stop.read().splitlines()
#add = open(r'wordlist/address.txt','r',encoding='utf-8')
#add_l = add.read().splitlines()
##location.extend(ner_list)
#location.extend(add_l)
#location.extend(['email','address','place','page','pages','name','soon','must','give',
#'order','date','type','location','please','refer','saving','read','blue','cross','shield','association','anthem','cobra','lmi'])
#location=[word.lower() for word in location]
#location =list(set(location))
#

################################################################################
#Text Preprocessing




def tokenize_clean(text):
    '''Using this function for removing the punctuation and short words as the first step
    
    '''
    global stop
    words2 = word_tokenize(text.lower())
    #words should not start with punctuation and the length of the word should be greater than 2
    words2 = [word for word in words2 if not (word.startswith("'") and len(word) <=2)]
    words2 = [word for word in words2 if word not in string.punctuation]
    ptext = ' '.join(words2)
    return ptext
    
def remove_cid(text):
    '''remove cid( fancy bullet encoding from the text)'''
    return text.replace('cid','')

def remove_numbers(text):
    '''remove the numbers as they are not required'''
    return re.sub(r'\d+','',text)   
    

def remove_indiv_punctuation(text):
    words2= text.split(' ')
    words2 = [word for word in words2 if not all(char in string.punctuation for char in word)]    
    ptext= ' '.join(words2)
    return ptext

def remove_apostro(text):
    words2 = text.split("'")
    words2 = [word for word in words2 if not all(char in string.punctuation for char in word)]    
    ptext= ' '.join(words2)
    return ptext
    
    
def remove_ordinal(text):
    for c in text:
        if ord(c) > 127:
            text.replace(c,'')
    return text  
    
def remove_url(text):
    '''remove all kinds of webpage addresses
    foo@demo.net	bar.ba@test.co.uk
    www.demo.com	http://foo.co.uk/
    http://regexr.com/foo.html?q=bar
    https://mediatemple.net
    '''
    return re.sub(r'((ht|f)tp(s)?:\/\/)?(w{0,3}\.)?[a-zA-Z0-9_\-\.\:\#\/\~\}]+(\.[a-zA-Z]{1,4})(\/[a-zA-Z0-9_\-\.\:\#\/\~\}]*)?','',text)


def remove_slash(text):
    words2 = text.split('/')
    words2 = [word for word in words2 if not all(char in string.punctuation for char in word)]    
    ptext= ' '.join(words2)
    words2 = ptext.split('\\')
    words2 = [word for word in words2 if not all(char in string.punctuation for char in word)]
    ctext = ' '.join(words2)
    return ctext

    
    
def remove_garbage(text):
    atext= text.replace('\x0c','')
    dtext=atext.replace('§','')
    ftext=dtext.replace('™','')
    gtext=ftext.replace('½','')
    htext=gtext.replace('®','')
    itext=htext.replace('\u00a9',' ')
    jtext = itext.replace(' – ','')
    ktext=jtext.replace('“',' ')    
    ltext=ktext.replace('”',' ')
    mtext=ltext.replace('’',' ')
    ntext=mtext.replace('  ',' ')
    otext=ntext.replace('÷','')
    ptext= otext.replace('0xa7','')
    qtext= ptext.replace('\x03','')
    rtext=qtext.replace('\x11','')
    stext=rtext.replace('vhh\x03','')
    ttext= stext.replace('iurp\x03','')
    utext = ttext.replace('•','')
    vtext= utext.replace('.','')
    wtext= vtext.replace('*',' ').replace('ñ',' ').replace('é',' ')
    tokens = wtext.split()
    tokens = [x.strip() for x in tokens]
    final= ' '.join(tokens)
    return final
    
def replace_hypen(text):
    ptext=text.replace('-','_')
    return ptext
    
    
def remove_common(text):
    stop=['blue','cross','shield','association','government','inc.','pepsico','email','reason','used','make','amend','people','sure','such','much']
    words = text.split()
    cleaned_tokens= [w for w in words if w not in stop]
    ptext = ' '.join(cleaned_tokens)
    return ptext

    
def remove_small(text):
    words = text.split()
    words2 = [word for word in words if not len(word) <=3]
    ptext= ' '.join(words2)
    return ptext

def state_abbr(text):
    '''Returns the string after expanding the states abbreviations'''
    return replace_words(text,WL_abbr_states)
    
    
def month_abbr(text):
    '''Takes string input and returns string with expanded month names,
    jan->january etc. Also accounts for multiple abbreviations for a month like
    sept or sep for september'''
    return replace_words(text,WL_month_abbrv)   
    
    
def remove_location(text):
    global location
    words = text.split()
    cleaned_tokens= [w for w in words if w not in location]
    ptext= ' '.join(cleaned_tokens)
    return ptext

     
    
def remove_wrong_words(text):
    import enchant
    d= enchant.Dict("en_US")
    text=text.lower()
    lines=text.split()
    lines=[word for word in lines if d.check(word)]
    return ' '.join(lines)  

def remove_dot(text):
    return text.replace('…','')

    

wnl = WordNetLemmatizer()

def lemmatize(ambiguous_word, pos=None, neverstem=False, 
              lemmatizer=wnl):
    """
    Tries to convert a surface word into lemma, and if lemmatize word is not in
    wordnet then try and convert surface word into its stem.

    This is to handle the case where users input a surface word as an ambiguous 
    word and the surface word is a not a lemma.
    """
    if pos:
        lemma = lemmatizer.lemmatize(ambiguous_word, pos=pos)
    else:
        lemma = lemmatizer.lemmatize(ambiguous_word)
   
    return lemma


def penn2morphy(penntag, returnNone=False):
    '''
    Used to get the pos tags that can be passed to 
    wn.lemmatize() as the original tags cannot be passed
    '''
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

def lemmatize_sentence(text, neverstem=False, keepWordPOS=False, 
                       tokenizer=word_tokenize, postagger=pos_tag, 
                       lemmatizer=wnl):
    words, lemmas, poss = [], [], []
    for word, pos in postagger(tokenizer(text)):
        if word not in WL_lemma_ignore:
            pos = penn2morphy(pos)
            lemmas.append(lemmatize(word.lower(), pos, neverstem,
                                lemmatizer))
            poss.append(pos)
            words.append(word)
        else:
            lemmas.append(word.lower())
            words.appemd(word.lower())
    if keepWordPOS:
        return words, lemmas, [None if i == '' else i for i in poss]
    return ' '.join(lemmas)

def change_words(text):
    text=replace_words(text,WL_w_ngrams)
    return text
    
def remove_stop(text):
    tokens=word_tokenize(text)
    tokens=[token for token in tokens if token not in stop]
    text = ' '.join(tokens)
    return text
    
    
def remove_underscore(text):
    text = re.sub(r'(^|\s)_(\S)',r'\1\2',text)
    return re.sub(r'(\S)_($|\s)',r'\1\2',text)


def remove_simlutaneous(text):
    tokens = word_tokenize(text)
    ptext = ' '.join([x[0] for x in groupby(tokens)])
    return ptext

def get_bigrams(text):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)

    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, 100)

    for bigram_tuple in bigrams:
        x = "%s_%s" % bigram_tuple
        tokens.append(x)

    result = ' '.join(tokens)
    return result
    
    
RULES_IN_ORDER = [
    remove_ordinal,
    tokenize_clean,
    remove_cid,
    remove_numbers,
    remove_url,
    remove_slash,
    remove_garbage,
    replace_hypen,
    #rm_control_chars,
    remove_underscore,
    remove_wrong_words,
    lemmatize_sentence,
    change_words,
    remove_small,
    remove_simlutaneous,
    change_words,
    remove_location,
    remove_dot,
    change_words,
    remove_simlutaneous,
    remove_stop,
    remove_simlutaneous
]

RULES_FOR_USER = [
    remove_ordinal,
    tokenize_clean,
    remove_cid,
    remove_numbers,
    remove_url,
    remove_slash,
    remove_garbage,
    replace_hypen,
    #rm_control_chars,
    remove_underscore,
    remove_wrong_words,
    lemmatize_sentence,
    change_words,
    remove_small,
    remove_simlutaneous,
    change_words,
    remove_location,
    remove_dot,
    change_words,
    remove_simlutaneous,
    remove_stop,
    remove_simlutaneous,
    get_bigrams
]



    
def normalize_text(text):
    """Preprocess the text document (string),
    as described in comments in function body

    Args:
        text: string to be processed
    Returns:
        Processed text
    """
    ptext = text
    for rule in RULES_IN_ORDER:
        ptext = rule( ptext )
        stext = replace_words(ptext,WL_w_ngrams)
    return stext


def normalize_user(text):
    """Preprocess the text document (string),
    as described in comments in function body

    Args:
        text: string to be processed
    Returns:
        Processed text"""
    ptext=text
    for rule in RULES_FOR_USER:
        ptext = rule(ptext)
    return ptext



    
if __name__ == "__main__":
    
 
    print('New Cleaner')
#    raw_data = r'raw_data'
    
    
#    import datetime
#    import csv
#    csvReader = csv.reader(open(r'new_to_clean.csv','r',encoding='utf-8'),delimiter=',',quotechar='"')
#    headers = next(csvReader)
#    data = [dict(zip(headers, row)) for row in csvReader]
    
#    for row in data:
#        id=row.get('id')
#        docs=row.get('data')
#        name=row.get('RecoveryEligible')

        
    
        
        
#    a=[d['data'] for d in data]  
#    b=[d['id']for d in data]
#    c=[d['RecoveryEligible'] for d in data]
    
    
#    name=[]
#    for x in c:
#        name.append(x)
    
#    len_docs = len(a)
#    norm=[]
    

#    cou=1
#    import time
#    for s in a:
#        stime= time.time()
#        print('starting doc: '+str(cou))
#        norm.append(normalize_text(s.strip()))
#        print('done: '+str(cou)+' \n total time:'+str((round(time.time()-stime,3))))
        #print('Took '+str(round(time.time()-stime),2) +' sec to clean doc:' +str(count))
#        cou=cou+1
    
        
#    new_file = '_'.join(['preprocessed',str(datetime.datetime.now().date())])    
#    new_file =os.path.join(raw_data, new_file+'.csv') 
#    header=['id','recoveryEligible','data']
#    with open(new_file, 'w',newline='',encoding='utf-8') as f:
#        writer = csv.writer(f)
#        writer.writerow(header)
#        writer.writerows(zip(b,name,norm))
#    f.close()
    
    
    
    
    
    
    