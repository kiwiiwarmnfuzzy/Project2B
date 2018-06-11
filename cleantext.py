#!/usr/bin/env python

"""Clean comment text for easier parsing."""

from __future__ import print_function

import re
import string
import argparse
import sys 
import json
import html
__author__ = ""
__email__ = ""

# Some useful data.
_CONTRACTIONS = {
    "tis": "'tis",
    "aint": "ain't",
    "amnt": "amn't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hell": "he'll",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "its": "it's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "ol": "'ol",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "somebodys": "somebody's",
    "someones": "someone's",
    "somethings": "something's",
    "thatll": "that'll",
    "thats": "that's",
    "thatd": "that'd",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "wed've",
    "well": "we'll",
    "were": "we're",
    "weve": "we've",
    "werent": "weren't",
    "whatd": "what'd",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "whod've",
    "wholl": "who'll",
    "whore": "who're",
    "whos": "who's",
    "whove": "who've",
    "whyd": "why'd",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

# You may need to write regular expressions.

CONTRACTIONS = {v: k for k, v in _CONTRACTIONS.items()}
contractions = re.compile(r'\b(' + '|'.join(CONTRACTIONS.keys()) + r')\b')
puncts = re.compile("(\w+(?:(-|'|/|’|\"|—|\)|\(|/)\w+)+|\w+|[.!?,;:$%\'“”@#]|\s)") 
#puncts = re.compile("(\w+(?:(-|'|/|’|—|\)|\(|/)\w+)+|\w+|[.!?,;:$%\'“”'\"#&\'+-/<=>@[\\]_`{|}~']|\s)")

def sanitize(text):
    """Do parse the text in variable "text" according to the spec, and return
    a LIST containing FOUR strings 
    1. The parsed text.
    2. The unigrams
    3. The bigrams
    4. The trigrams
    """

    # YOUR CODE GOES BELOW:

    # 1. replace new lines and tab characters with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 2. remove urls, replace them with the empty string
    # match url like [url](actual url)
    text = re.sub(r'\[(.*?)\]\(http[s]?://(?:[a-zA-Z]|[0-9]|[$#-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\)', r'\1', text)
    # match url that is just plain url
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$#-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # match internal reddit links
    text = re.sub(r'\[(.*?)\]\((?:[a-zA-Z]|[0-9]|[$#-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\)', r'\1', text)
    text = re.sub(r' \/(?:[a-zA-Z]|[0-9]|[$#-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\/(?:[a-zA-Z]|[0-9]|[$#-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # 5. remove all punctuation (including special characters that are
    # not technically punctuation) except .!?,;:
    lst = puncts.findall(text)#html.unescape(text))
    text = ''
    for i in lst:
        text += i[0]

    # 4. separate all external punctuations:
    # .!?,;:
    # into a single token

    # but only if they are not in the middle of a word
    text = re.sub(r'([!?:])', r' \1 ', text)
    
    # delimiters that may appear in the middle of the word: .,;
    text = re.sub(r'(\w|%|$)(\.|,|;)$', r'\1 \2 ', text) # at the end of line
    text = re.sub(r'(\w|%|$)(\.|,|;) ', r'\1 \2 ', text) # at the end of sentence but not line

    # deal with elipses:
    # identify elipses: .{2,} followed by space or end of line and preceded by a word, ($%)
    # separate them into a token
    text = re.sub(r'((\w)(\.+) )', r'\2 \3 ', text) 
    text = re.sub(r'((\w)(\.+$))', r'\2 \3 ', text)
    # split the .{2,} into '. '{2,}
    text = re.sub(r' (\.+) ', lambda m: m.group(1).replace('.', ' . '), text)

    # 6. convert all text to lower case
    text = text.lower()
    
    # 3. split text on a single space
    parts = text.split()

    parsed_text = ''
    unigrams = ''
    bigrams = ''
    trigrams = ''
    
    def isdelim(s):
        return (s=='.') or (s=='!') or (s=='?') or (s==',') or (s==';') or (s==':')

    for i, token in enumerate(parts):
        parsed_text += (token+' ')
        if (not isdelim(token)):
            unigrams += (token+' ')

    i = 0
    while(i < len(parts)-1):
        if (isdelim(parts[i])):
            i += 1
        elif (not isdelim(parts[i+1])):
            bigrams += (parts[i]+"_"+parts[i+1]+" ")
            i += 1
        else:
            i += 2
    
    i = 0
    while(i < len(parts)-2):
        if (isdelim(parts[i])):
            i += 1
        elif (not isdelim(parts[i+2])):
            if (not isdelim(parts[i+1])):
                trigrams += (parts[i]+"_"+parts[i+1]+"_"+parts[i+2]+" ")
                i += 1
            else:
                i += 2
        else:
            i += 3

    return [parsed_text[:-1], unigrams[:-1], bigrams[:-1], trigrams[:-1]]


if __name__ == "__main__":
    # This is the Python main function.
    # You should be able to run
    # python cleantext.py <filename>
    # and this "main" function will open the file,
    # read it line by line, extract the proper value from the JSON,
    # pass to "sanitize" and print the result as a list.

    # YOUR CODE GOES BELOW.
    if (len(sys.argv) < 2):
        print("Please enter the name of the json file")
        exit(1)
    with open(sys.argv[1], encoding='utf-8') as f:
        for line in f:
            s = sanitize(json.loads(line)["body"])
            print(s)
            
