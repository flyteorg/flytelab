import pandas as pd
import nltk
import re
import swifter
from nltk.tokenize import word_tokenize
from string import punctuation
from unidecode import unidecode
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator


def lower_text(text: str) -> str:
        """
        Method responsable for lower text
        Args:
            text (str): text to be lower
        Return:
            low text
        """
        return text.lower()
 
def clean_text(texts: str) -> str:
    """
    Method responsable remove unnecessary parts of the text
    Args:
        text (str): text 
    Return:
        text without unnecessary parts
    """
    
    #remove empty line
    clean_empt_msg = re.compile('\n\s*\n')
    text = re.sub(clean_empt_msg, " ", texts)

    text = unidecode(text)

    #Remove API mensage
    clean_msg = re.compile('\(.*?\)')
    text = re.sub(clean_msg, ' ', text)
    
    #Remove html characteres
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(cleanr, ' ', text)
     
    # Remove punctuations and numbers
    clean_pontuation = re.compile('[^a-zA-Z]')
    text = re.sub(clean_pontuation, ' ', text)
    
    # Single character removal
    clean_char = re.compile(r"\s+[a-zA-Z]\s+")
    text = re.sub(clean_char, ' ', text)
    
    # Removing multiple spaces
    clean_space = re.compile(r'\s+')
    text = re.sub(clean_space, ' ', text)
    
    return text


def remove_stopwords(list_tokens:list,stopword=stopwords)->str:
    """
    Method responsable for removing stopwords of the text
    Args:
        list_tokens (list): list of tokens of a text
    Return:
        text without stopwords
    """
    stopword = stopword.words('portuguese') + list(punctuation)+["\n",'municipio','clima']
    txt_wo_stopwords = filter(lambda item: item not in stopword, list_tokens)
    return " ".join(txt_wo_stopwords)


def tokenizer(text:str)->list:
    """
    Method responsable for tokenizing the text
    Args:
        text (str): text to be tokenized
    Return:
        list of words (str)
    """
    return word_tokenize(text)

def preprocess_text(dataframe: pd.DataFrame,column_name:str) -> None:
    """
    Method responsable for doing all pre-process methods
    Args:
        dataframe (pd.DataFrame): Serie with all texts
    Return:
        pd.DataFrame with all pre-processed
    """
    aux = dataframe[column_name].str.lower()
    aux = aux.swifter.apply(lambda x: clean_text(str(x)))
    aux = aux.swifter.apply(lambda x: remove_stopwords(list_tokens=tokenizer(x)))        
    return aux

def translate_description_series(dataframe:pd.DataFrame,column_name:str,
                                 target_lang: str='pt')->pd.Series:
    """
    Method responsable for translate non-protuguese texts
    Args:
        series_text (pd.Series): Text Series
        pre_trained_path (str): path to pre-trained language identifier
        columns_name (str): columns to be translated
        id_columns_name (str): hotel sku column name
    Return:
        pd.DataFrame with all text (translated e non-translated)
    """    
    dataframe[column_name] = dataframe[column_name].fillna("")
    dataframe[column_name] = dataframe[column_name].swifter.apply(lambda x: translate_description(x,target_lang) 
                                                                                if type(x)==str else x)
    return dataframe[column_name]

def translate_description(text:str, target_lang: str='pt') -> str:
    """
    Method responsable for translate non-portuguese text
    Args:
        text (str): text to be translated
    Return:
        pd.DataFrame with all text (translated e non-translated)
    """
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return text
