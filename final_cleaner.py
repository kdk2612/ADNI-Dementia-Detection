#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 23:50:30 2017

@author: karankothari
"""

#!/usr/bin/python2
import re
from datetime import date
from nltk.stem import WordNetLemmatizer
from sys import path
DIR_WORDLIST='wordlist'
################################################################################
# Wordlists for replacement

def read_wordlist(wl):
    #return dict( [ l.strip().split(',')
    return [ tuple(l.strip().split(','))
             for l in open('%s/%s'%(DIR_WORDLIST,wl), 'r') ] #)

WL_abbrv_medical  = read_wordlist('abbrv_medical')
WL_change_phrases = read_wordlist('change_phrases')
WL_dementia_terms = read_wordlist('dementia_terms')
WL_numerals       = read_wordlist('numerals')
WL_months         = read_wordlist('months')
WL_months_index   = dict(read_wordlist('months_index'))  # This one is special
WL_mwe_medical    = read_wordlist('mwe_medical')
WL_mwe_ngrams     = read_wordlist('mwe_from_ngrams')
WL_stop           = read_wordlist('stop')

WL_lemma_ignore   = read_wordlist('lemmatize_ignore')


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


################################################################################
# Text Pre-processing

def pad(text):
    return ' ' + text + ' '

def unpad(text):
    return text.strip()

def remove_punct(text):
    # First one is for contractions and possessives (don't, alzheimer's, etc.)
    ptext = re.sub( r"([A-Za-z])'([A-Za-z])", r"\1\2", text)
    return re.sub( '[\.,!?"\':;\(\)\{\}\[\]$%&+*#@\^~_<>\|]', ' ', ptext )

def multispace(text):
    return ' '.join(text.split())

def lowercase(text):
    return text.lower()

def stoplist(text):
    return replace_words(text, WL_stop)

def insert_spaces(text):
    """Space out typos where words and numbers are pushed together,
    e.g. '1998surgery'"""
    ptext = re.sub( r'([a-z])([0-9])', r'\1 \2', text )
    return re.sub( r'([0-9])([a-z])', r'\1 \2', ptext )

def unhyphen(text):
    return re.sub( r'([^ \-]*)\-([^ \-]*)',
                   r'\1 - \2',
                   text )

def rehyphen(text):
    return re.sub( r'([^ ]*)\s+\-\s+([^ ]*)', r'\1-\2', text )

def dehyphen(text):
    return re.sub( r'\-', r' ', text )

def deslash(text):
    """Remove slashes between letters/words"""
    return re.sub( r'([a-z ])/([a-z ])', r'\1 \2', text )


def lemmatize(text):
    """Use NLTK WordNetLemmatizer"""
    wnl = WordNetLemmatizer()
    return ' '.join([ wnl.lemmatize(w) for w in text.split() if w not in WL_lemma_ignore ])


################################################################################
# Replacement

#RE_compound_numerals = re.compile( r'(' )

def numerals2arabic(text):
    return replace_words(text, WL_numerals)

def roman2arabic(text):
    ## unimplemented
    return text

def remove_remaining_numbers(text):
    """Remove any numbers or fractions that may still be present"""
    return re.sub( r'\b([0-9]+[/]?|[/]?[0-9]+)\b', r' ', text )

def remove_dates(text):
    """Remove date tags entirely""" 
    abc=re.sub( r'DATE_[0-9]{4}_[0-9]{2}_[0-9]{2}', r'', text )
    #print(abc)
    return abc

def bin_age_35_65(text):
    """Bin AGE tags into <35, [35-65), and >=65"""
    return bin_age_tags(text)

def bin_age_40_90_by10s(text):
    """Bin AGE tags into <40, 40-50, 50-60, 60-70, 80-90, and >=90"""
    return bin_age_tags(text)

def remove_change_phrases(text):
    """Remove phrases like "no changes" or "remains stable" """
    return replace_words(text, WL_change_phrases)

def remove_dementia_terms(text):
    """Remove dementia terms like 'alzheimers disease' or 'normal_control' """
    return replace_words(text, WL_dementia_terms)


################################################################################
# Expansion

def expand_abbrv_medical(text):
    return replace_words(text, WL_abbrv_medical)

def expand_abbrv_months(text):
    return replace_words(text, WL_months)

def expand_range(text):
    return re.sub( r'([0-9]+)[ ]?(\-|to)[ ]?([0-9]+) (year|month|week|day|time)[s]?',
                   r'\1 \4 to \3 \4', text )

def expand_grade(text):
    return re.sub( r'grade ([0-9]+)[ ]?(\-|to)[ ]?([0-9]+)',
                   r'grade_\1 to grade_\3', text )

def expand_years(text):
    return re.sub( r'([0-9]{4})-([0-9]{4})',
                   r'\1 to \2', text )

def expand_rate(text):
    return re.sub( r'/(night|day|week|wk|month|mo|year|yr)', r' per \1', text )


################################################################################
# Compression 

def compress_mwe_medical(text):
    return replace_words(text, WL_mwe_medical)

def compress_mwe_ngrams(text):
    return replace_words(text, WL_mwe_ngrams)


################################################################################
# Age Tagging

AGE_TAG_STRING = 'AGE_%03d_%02d'

RE_age_1 = re.compile( r'((?:([0-9]+)[\- ]?(1\/2|half)?[\- ]?(?:year|yr)[s]?)?' +\
                       r'[\- ]?(?:([0-9]+)[\- ]month[s]?)?' +\
                       r'[\- ]?(?:([0-9]+)[\- ]week[s]?)?' +\
                       r'[\- ]?(?:([0-9]+)[\- ]day[s]?)?[\- ]old)' )
RE_age_2 = re.compile( r'(([0-9]{1,3})[ ]?y[/]?o)' )

def tag_age_1(text):
    """Tag age strings

    This is capable of extracting years, half-years, months, weeks, and days,
    but we only include the first three.

    Tags are of the form 'AGE_<years>_<months>' with carryover.

    e.g.    '10 year 1-month old' => 'AGE_10_1'
            '10-1/2 year-old'     => 'AGE_10_6'
            '5 month 3 day old'   => 'AGE_0_5'
            '85-month-old'        => 'AGE_7_1'

    NOTE:
        !!!
        I found a major bug in that regex which is that it will match the string
        
        " old"
        
        wih no numbers at all. Wow. And I don't feel like fixing the regex so I just
        added a condition to check if that happened and skip if so. Ta-da!
    """
    ptext = text
    for age in RE_age_1.findall(ptext):
        ## See docstring NOTE
        if age[0] == " old": continue
        y  = 0 if age[1] == '' else int(age[1])
        m  = 0 if age[3] == '' else int(age[3])
        if age[2] != '': # half years
            m += 6
        ptext = ptext.replace( age[0], AGE_TAG_STRING % (y+m/12, m%12), 1 )
    return ptext

def tag_age_2(text):
    """Tag additional age cases, easier to just do separate.

    Same result as other age function.

    e.g.    '75 y/o' => 'AGE_75_0'
            '64 yo'  => 'AGE_64_0'
    """
    ptext = text
    for age in RE_age_2.findall(ptext):
        ptext = ptext.replace(age[0], AGE_TAG_STRING % (int(age[1]), 0), 1)
    return ptext


################################################################################
# Date Tagging
#
# ***
# See correpsonding function (numbered) for explanation/examples of each regex.
# The numbers may correspond to order to don't rearrange them without checking.
# ***

# SPACES ARE A TEMPORARY FIX FOR SPACING THING IN THE OVERCOMPLICATED REGEX
# BUT IT'S REALLY NOT AS BIG OF A DEAL AS THE CAPSLOCK WOULD HAVE YOU BELIVE
DATE_TAG_STRING = ' DATE_%04d_%02d_%02d '

CURR_CENTURY = date.today().year / 100 * 100
LAST_CENTURY = CURR_CENTURY - 100
CURR_YEAR    = date.today().year % 100

ALL_DIGITS = [ str(x) for x in range(10) ]
ALL_BLANKS = [ 'unknown', 'unk', 'uk', '-', '--', '----' ]
ALL_MONTHS = WL_months_index.keys()

# This is not grouped here because some regexes may want to ignore or expand it
PIPED_MONTHS = r'%s' % ('|'.join(ALL_MONTHS))
PIPED_ALL    = r'[0-9]{1,2}|%s|%s' % ( PIPED_MONTHS, '|'.join(ALL_BLANKS) )

# ***
# SEE FUNCTIONS FOR EXPLANATIONS AND EXAMPLES
# ***


RE_date_1 = re.compile( r'((' + PIPED_ALL + r')/(' + PIPED_ALL +
                        r')/([0-9]{2,4}|\-\-|\-\-\-\-))' )
RE_date_2 = re.compile( r'(?:[^0-9]?)((' + PIPED_ALL + r')[/\-]([0-9]{4}))' )
RE_date_3 = re.compile( r'(([0-9]{1,2}) (' + PIPED_MONTHS + r') ([0-9]{4}))'  )
RE_date_4 = re.compile( r'(('+ PIPED_MONTHS + r')[ ]?(?:([0-9]{4}' +\
                        r'|[0-9]{1,2})(?:st|nd|rd|th)?(?:[ ]?([0-9]{4})?)))' )
RE_date_5 = re.compile( r'(?:[^0-9]|^)(?:(?:((?:[0-9]{4})([-.])' +\
                        r'(?:[0-9]{1,2})([-.])(?:[0-9]{1,2})))' +\
                        r'|((?:[0-9]{1,2})([-.])(?:[0-9]{1,2})([-.])' +\
                        r'(?:[0-9]{2,4})))(?:[^0-9]|$)' )
RE_date_6 = re.compile( r'(([^_])(20[0-9]{2}|190[1-9]|19[1-9][0-9]))' ) # Lazy, depends on padded spaces


def tag_date_1(text):
    """Tag the slash formatted dates with 3 fields

    Tags are of the form 'DATE_<year>_<month>_<day>' with leading zeros.

    Handles American format with numbers, 2 digit or 4 digit year:

    e.g.    '2/3/2013' => 'DATE_2013_02_03'
            '02/03/13' => 'DATE_2013_02_03'

    Spelled out months in either format, with abbreviations:

    e.g.    'feb/03/2013'      => 'DATE_2013_02_03'
            'february/03/2013' => 'DATE_2013_02_03'
            '03/feb/2013'      => 'DATE_2013_02_03'
            '03/february/2013' => 'DATE_2013_02_03'

    Specified blank values are replaced with 0's:

    e.g.    '--/03/2013'       => 'DATE_2013_00_03'
            '02/--/2013'       => 'DATE_2013_02_00'
            '--/--/2013'       => 'DATE_2013_00_00'
            '--/--/----'       => 'DATE_0000_00_00'
            '--/--/--'         => 'DATE_0000_00_00'
            'feb/unk/2013'     => 'DATE_2013_02_00'
            'feb/unknown/2013' => 'DATE_2013_02_00'

    Also assumes that any two-digit year that is greater than the current year
    by more than 10 is from the previous century, and < from this century.
    This is to allow for references to future events.

    e.g. if the current year is 2014, then < 19 is considered 20__

            '12/31/16'   => 'DATE_2016_12_31'  # Future event
            '12/31/87'   => 'DATE_1987_12_31'  # Distant past event
            '12/31/09'   => 'DATE_2009_12_31'  # Recent event
    """
    ptext = text
    for date in RE_date_1.findall(ptext):
        # If year is blank then so is everythign else
        if date[3] in ALL_BLANKS:               # --/--/--, --/--/----
            y, m, d = 0, 0, 0
        else:
            # Find year
            y = int(date[3])
            if len(date[3]) == 2:
                y = int(date[3]) + (CURR_CENTURY if y < CURR_YEAR+10 else LAST_CENTURY)
            elif len(date[3]) == 3:
                continue

            # Find month and day
            if date[1][0] in ALL_DIGITS:
                if date[2][0] in ALL_DIGITS:    # 10/10/2010, etc. (Assume American)
                    m = int(date[1])
                    d = int(date[2])
                elif date[2] in ALL_BLANKS:     # 10/unk/2010, 10/--/2010, etc
                    m = int(date[1])
                    d = 0
                elif date[2] in ALL_MONTHS:     # 10/oct/2010, etc
                    m = int(WL_months_index[date[2]])
                    d = int(date[1])
                else:                           # Bad match
                    continue
            elif date[1] in ALL_BLANKS:
                d = 0
                if date[2] in ALL_BLANKS:       # --/--/2010, unk/unk/2010, etc
                    m = 0
                elif date[2][0] in ALL_DIGITS:  # --/10/2010, unk/10/2010, etc
                    m = int(date[2])
                elif date[2] in ALL_MONTHS:     # --/oct/2010, unk/oct/2010, etc
                    m = int(WL_months_index[date[2]])
                else:                           # Bad match
                    continue
            elif date[1] in ALL_MONTHS:
                m = int(WL_months_index[date[1]])
                if date[2] in ALL_BLANKS:       # oct/--/2010, oct/unk.2010, etc
                    d = 0
                elif date[2][0] in ALL_DIGITS:  # oct/10/2010
                    d = int(date[2])
                else:                           # Bad match
                    continue
            else:                               # Bad match
                continue
        ptext = ptext.replace( date[0], DATE_TAG_STRING%(y,m,d), 1 )
    return ptext

def tag_date_2(text):
    """Tag 2-field numeric slash formatted dates
    """
    ptext = text
    for date in RE_date_2.findall(ptext):
        d = 0
        y = int(date[2])
        if date[1] in ALL_BLANKS:       # --/2010, unk/2010, ----/2010
            m = 0
        elif date[1][0] in ALL_DIGITS:  # 10/2010, 1/2010
            m = int(date[1])
            if m > 12:                  # Bad match
                continue
        else:                           # Bad match
            continue
        ptext = ptext.replace( date[0], DATE_TAG_STRING%(y,m,d), 1 )
    return ptext

def tag_date_3(text):
    """Tag dates with worded months and non-American format

    Tags are of the form 'DATE_<year>_<month>_<day>' with leading zeros and
    bounds checking.

    This will get dates with day coming before month, requiring a 4 digit year
    and also checks bounds the day to avoid stupid matches.

    e.g.    '20 january 2001' => 'DATE_2001_01_20'
            '1.0 july 2003'   => *Skip*

    THIS SHOULD BE RUN BEFORE THE OTHER NAMED-MONTH TAGGER
    """
    ptext = text
    for date in RE_date_3.findall(ptext):
        d, m, y = int(date[1]), int(WL_months_index[date[2]]), int(date[3])
        if d < 1 or d > 31:
            continue
        ptext = ptext.replace( date[0], DATE_TAG_STRING%(y,m,d), 1 )
    return ptext

def tag_date_4(text):
    """Tag date string with named months in American format

    Tags are of the form 'DATE_<year>_<month>_<day>' with leading zeros and
    bounds checking.

    Must require either day, year, or both (i.e. no standalone months)

    e.g.    'january 2 2001'  => 'DATE_2001_01_02'
            'january 2001'    => 'DATE_2001_01_00'

    If no 4 digit year is provided, then the following assumptions are made:

        1. If single digit, then assume day

            e.g.    'january 2'  => 'DATE_0000_01_02'

        2. If 2 digits with leading 0, then assume shorthand for year 20__

            e.g.    'january 02' => 'DATE_2002_01_00'

        3. If two digits >= 10 and <= 31, assume day

            e.g.    'january 23' => 'DATE_0000_01_23'

        4. If two digits > 31, assume shorthand for year 19__

            e.g.    'january 95' => 'DATE_1995_01_00'

    WARNING:
        Since unspecified years are allowed, it may be useful to add processing
        to infer the year based on the year of the record, e.g. 'january 5'
        mentioned in a record from February of 1998 likely refers to the same
        year, but if it said 'december 5' then it was likely referring to the
        previous year.
    """
    ptext = text
    for date in RE_date_4.findall(ptext):
        if date[3] == '':
            if len(date[2]) == 4:               # jan 2003
                y = int(date[2])
                d = 0
            elif len(date[2]) == 1:             # jan 2
                y = 0
                d = int(date[2])
            else:
                if date[2].startswith('0'):     # jan 03 => jan 2003
                    y = 2000 + int(date[2])
                    d = 0
                else:
                    d = int(date[2])
                    if d > 31:                  # jan 95 => jan 1995
                        y = 1900 + int(date[2])
                    elif d >= 10:               # jan 21 => 21st of jan
                        y = 0
                    else:
                        # <10 already handled
                        continue                # Bad match
        else:
            y = int(date[3])
            d = int(date[2])
        # Month is gauranteed
        m = int(WL_months_index[date[1]])
        ptext = ptext.replace( date[0], DATE_TAG_STRING%(y,m,d), 1 )
    return ptext

def tag_date_5(text):
    """Tag date string with dashes or dots

    Tags are of the form 'DATE_<year>_<month>_<day>' with leading zeros with
    bounds checking.

    This will handle either American format or year->month->day format, and
    handles single or double digits for days and months

    e.g.    '03-03-2003'    => 'DATE_2003_03_03'
            '2003-03-03'    => 'DATE_2003_03_03'
            '3.3.2003'      => 'DATE_2003_03_03'
    
    If first and last fields are both 2 digits, then American format.
    Century is assumed to be current if 2-digit year would be less than 10 years
    from now, and last century otherwise:

    e.g.    If the current year is 2014:
    
            '03-04-05'      => 'DATE_2005_03_04'
            '03-04-20'      => 'DATE_2020_03_04'
            '03-04-80'      => 'DATE_1980_03_04'

    The regex will match either '-' or '.' but this function will ignore
    mismatched delmiters:

    e.g.    '3.4-2003'      => *Skip*

    Bounds checking:

    e.g.    '4-0-3000       => *Skip*
    """
    ptext = text
    for date in RE_date_5.findall(ptext):
        if date[0] != '' and date[1] == date[2]:                    # 2013-03-03
            y, m, d = [ int(f) for f in date[0].split(date[1]) ]
            repl = date[0]
        elif date[3] != '' and date[4] == date[5]:                  # American
            m, d, y = [ int(f) for f in date[3].split(date[4]) ]
            if 0 < y < 100:                                         # 03-03-03
                y += CURR_CENTURY if y < CURR_YEAR+10 else LAST_CENTURY
            repl = date[3]
        else:
            continue                                                # Bad match

        # Bounds check
        if d < 1 or d > 31 \
           or m < 1 or m > 12 \
           or y > CURR_CENTURY+CURR_YEAR+20 or y < CURR_CENTURY+CURR_YEAR-110:
            continue

        ptext = ptext.replace( repl, DATE_TAG_STRING%(y,m,d), 1 )
    return ptext

def tag_date_6(text):
    """Tag standalone years between 1901 and 2099.

    "1900" is avoided because it is used for "7:00 pm" sometimes and nobody
    is alive from then anyway so it wouldn't show up in a record.
    The point was to minimize collateral damage of the normalization, and as far
    as I can tell, there are no numbers in the dataset that will be matched
    incorrectly here.

    e.g.    '1993' => 'DATE_1993_00_00'
            '2013' => 'DATE_2013_00_00'

    NOTE:
        I am being super lazy about this one because it's being a huge pain.
    """
    ptext = text
    for date in RE_date_6.findall(ptext):
        ptext = ptext.replace( date[0], date[1] + DATE_TAG_STRING%(int(date[2]),0,0), 1)
    return ptext


####################################################
def bin_age_tags(text):
    words= text.split()
    for x in range(len(words)):
        if words[x].startswith('AGE_'):
            n=words[x].split('_')[-2]
            n=bins(n)
            words[x]= 'AGE_'+n
    return ' '.join(words)


def bins(n):
    if n in ['010','011','012','013','014','015','016','017','018','019']:
        return '>=10_<20'
    if n in ['020','021','022','023','024','025','026','027','028','029']:
        return '>=20_<30'
    if n in ['030','031','032','033','034','035','036','037','038','039']:
        return '>=30_<40'
    if n in ['040','041','042','043','044','045','046','047','048','049']:
        return '>=40_<50'
    if n in ['050','051','052','053','054','055','056','057','058','059']:
        return '>=50_<60'
    if n in ['060','061','062','063','064','065','066','067','068','069']:
        return '>=60_<70'
    if n in ['070','071','072','073','074','075','076','077','078','079']:
        return '>=70_<80'
    if n in ['080','081','082','083','084','085','086','087','088','089']:
        return '>=80_<90'
    if n in ['090','091','092','093','094','095','096','097','098','099']:
        return '>=90_<100'
    
####################################################


################################################################################
##
## THIS IS THE ORDER OF THE RULES/FUNCTIONS.
##
## THERE ARE DEPENDENCIES BETWEEN CERTAIN RULES.
##
################################################################################

RULES_IN_ORDER = [
    remove_punct,           # has to be at start
    multispace,             # has to be at start
    pad,
    lowercase,              # has to be at start
    insert_spaces,
    expand_grade,           # must come before UNHYPHEN
    expand_range,           # must come before UNHYPHEN
    expand_years,           # must come before DATE TAGGING
    expand_rate,
    unhyphen,
    numerals2arabic,        # depends on UNHYPHEN
    expand_abbrv_months, 
    expand_abbrv_medical,
    rehyphen,
    stoplist,               # has to be after MULTISPACE,LOWERCASE,PUNCT
    lemmatize,
    tag_age_1,              # depends on REHYPHN and NUMERALS
    tag_age_2,              # depends on REHYPHN and NUMERALS
    tag_date_1,
    tag_date_2,
    tag_date_3,
    tag_date_4,
    tag_date_5,
    tag_date_6,             # must come AFTER other date taggers
    compress_mwe_medical,
    compress_mwe_ngrams,
    remove_remaining_numbers,
    remove_dates,
    bin_age_40_90_by10s,
    dehyphen,
    deslash,
    remove_change_phrases,
    remove_dementia_terms,
    multispace,             # once more, just for fun
    unpad,
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
    return ptext

def normalize_user(text):
    ptext = text
    for rule in RULES_IN_ORDER:
        ptext = rule( ptext )
    return ptext


def run_clean():
    
    csvReader = csv.reader(open(r'C:\Users\kkothari\Desktop\text_mining\raw_data\preprocessed_2017-01-05.csv','r',encoding='utf-8'),delimiter=',',quotechar='"')
    headers = next(csvReader)
    data = [dict(zip(headers, row)) for row in csvReader]
    
    for row in data:
        id=row.get('id')
        docs=row.get('data')
        name=row.get('RecoveryEligible')

#        
#    
#        
        
    a=[d['data'] for d in data]  
    b=[d['id']for d in data]
    c=[d['recoveryEligible'] for d in data]
#    
    #df=df.dropna()
#    a=list(df['data'])
#    c=list(df['recoveryEligible'])
#    b=list(df['id'])
    name=[]
    for x in c:
        name.append(x)
    
    len_docs = len(a)
    norm=[]
    

    cou=1
    import time
    for s in a:
        stime= time.time()
        print('starting doc: '+str(cou))
        norm.append(normalize_text(s.strip()))
        print('done: '+str(cou)+' \n total time:'+str((round(time.time()-stime,3))))
        #print('Took '+str(round(time.time()-stime),2) +' sec to clean doc:' +str(count))
        cou=cou+1
        
    new_file = '_'.join(['preprocessed',str(datetime.datetime.now().date())])    
    new_file =os.path.join(raw_data, new_file+'.csv') 
    header=['id','recoveryEligible','data']
    with open(new_file, 'w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(b,c,norm))
    f.close()
        
    
    
if __name__ == "__main__":
    
#    import pandas as pd
#    df= pd.read_csv(r'C:\Users\kkothari\Desktop\text_mining\raw_data\preprocessed_2016-12-21.csv',delimiter=',',header='infer',encoding = "ISO-8859-1")
#    

    raw_data = r'raw_data'
    
    
    import datetime
    import csv

    