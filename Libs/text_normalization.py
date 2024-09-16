import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce
import spacy

# Step 1: Define a contraction mapping
contraction_mapping = {
    "ain't": "is not",
    "can't": "cannot",
    "can't've": "cannot have",
    "aren't": "are not",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    # custom one
    "pda":"push down automaton",
    "nfa":"non deterministic finite state automaton",
    "dfa":"deterministic finite state automaton",
    "cfg":"context free grammar",
    "ay":"accademic year",
    "ai":"artificial intelligence",
    "ects": "accademic credits",
    "cfu": "accademic credits",
    "credits": "accademic credits",
    "merged_columns": "",
    "teacher": "teach",
    "merged_column": ""
}

# Step 2: Function to expand contractions
def normalize_text(text, mapping, stemming = False, lemmatization = False):
    # lowercase
    text = text.lower()
    # remove noisy punctuation
    punctuation_to_remove = "!.:"
    translation_table = str.maketrans("", "", punctuation_to_remove)
    text = text.translate(translation_table)

    text = text.replace("\xa0", " ")
    text = text.replace('\n', " ")

    if stemming:
        ps = PorterStemmer()
        words = word_tokenize(text)
        # using reduce to apply stemmer to each word and join them back into a string
        text = reduce(lambda x, y: x + " " + ps.stem(y), words, "")

    # remove contracted forms
    pattern = re.compile(r'\b(' + '|'.join(mapping.keys()) + r')\b')
    def replace(match):
        key = match.group()
        if key in mapping:
            return mapping[key]
        else:
            return key  # Return the original key if not found in the mapping
    normalized_content = pattern.sub(replace, text)

    if lemmatization:
        # Load the spaCy English model
        nlp = spacy.load('en_core_web_sm')
        # Process the text using spaCy
        doc = nlp(normalized_content)
        # Extract lemmatized tokens
        lemmatized_tokens = [token.lemma_ for token in doc]
        # Join the lemmatized tokens into a sentence
        normalized_content = ' '.join(lemmatized_tokens)

    return normalized_content   