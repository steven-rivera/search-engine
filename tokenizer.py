from collections import defaultdict
import nltk


def tokenize(text: str) -> list[str]:
    """
    Takes a string and creates a list of "tokens"
    where each token is a string of alphanumeric characters whose
    len is greater than 1. 
    """
    
    if text == "" or text == None:
        return list()
    
    text = text.strip().lower() 
    tokens = nltk.tokenize.word_tokenize(text)
    
    return [token for token in tokens if isValidToken(token)] 


def isValidToken(token: str) -> bool: 
    """
    Token is valid if each character in the token is 
    a either a-z, or 0-9 and is not an empty string. Token 
    is not valid if it is only a single char and not the char 
    'a' or 'i'.
    """
    
    # ord('a') = 97, ord('z') = 122
    # ord('0') = 48, ord('9') = 57
    for char in token:
        unicodeValue = ord(char)
        if ( not (97 <= unicodeValue <= 122) ) and ( not (48 <= unicodeValue <= 57) ):
            return False
    

    if len(token) <= 1 and token not in {'a', 'i'}:
        return False

    return True


def computeWordFrequencies(tokens: list[str]) -> defaultdict[str, int]:
    """ 
    Counts the number of times the given token appears in the list 
    and returns a dictionary where the key is the token and the value is 
    the frequency.
    """
    wordCount = defaultdict(int)

    for token in tokens:
        wordCount[token] += 1

    return wordCount


def computeWordPositions(tokens: list[str]) -> defaultdict[str, list[int]]:
    """ 
    Returns a dictionary where the key is the token and the value is
    a list of positions where that token appeared. The positions starts 
    at 1.
    """
    positions = defaultdict(list)

    for position, token in enumerate(tokens, start=1):
        positions[token].append(position)

    return positions



try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
