"""Text preprocessing tools."""

# pylama: ignore=W0611
# pylint: disable=unused-import,broad-except

import re
from typing import List
from string import punctuation

import swifter
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
from deep_translator import GoogleTranslator


def lower_text(text: str) -> str:
    """Lower a text.

    Args:
        text (str): text to be lowered

    Returns:
        str: lower text
    """
    return text.lower()


def clean_text(texts: str) -> str:
    """Remove unnecessary parts of the text.

    Args:
        text (str): text to be cleaned

    Returns:
        str: cleaned text
    """
    # Remove empty lines
    clean_empt_msg = re.compile(r'\n\s*\n')
    text = re.sub(clean_empt_msg, " ", texts)

    # Transliterate into ASCII
    text = unidecode(text)

    # Remove API mensage
    clean_msg = re.compile(r'\(.*?\)')
    text = re.sub(clean_msg, ' ', text)

    # Remove HTML characteres
    cleanr = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(cleanr, ' ', text)

    # Remove punctuations and numbers
    clean_pontuation = re.compile(r'[^a-zA-Z]')
    text = re.sub(clean_pontuation, ' ', text)

    # Single character removal
    clean_char = re.compile(r"\s+[a-zA-Z]\s+")
    text = re.sub(clean_char, ' ', text)

    # Removing multiple spaces
    clean_space = re.compile(r'\s+')
    text = re.sub(clean_space, ' ', text)

    return text


def remove_stopwords(list_tokens: List[str],
                     stopword: List[str] = stopwords) -> str:
    """Remove stopwords of the text.

    Args:
        list_tokens (List[str]): list of sentence tokens

    Returns:
        List[str]: text without stopwords
    """
    stopword = (
        stopword.words('portuguese') +
        list(punctuation) +
        ["\n", 'municipio', 'clima']
    )

    txt_wo_stopwords = filter(lambda item: item not in stopword, list_tokens)
    return " ".join(txt_wo_stopwords)


def tokenizer(text: str) -> List[str]:
    """Tokenize the text.

    Args:
        text (str): text to be tokenized

    Returns:
        List[str]: list of sentence tokens
    """
    return word_tokenize(text)


def preprocess_text(dataframe: pd.DataFrame, column_name: str) -> pd.Series:
    """Execute all of the preprocess methods.

    Args:
        dataframe (pd.DataFrame): dataframe with column to be processed
        column_name (str): column name to be processed

    Returns:
        pd.Series: column processed
    """
    aux = dataframe[column_name].str.lower()
    aux = aux.swifter.apply(lambda x: clean_text(str(x)))
    aux = aux.swifter.apply(
        lambda x: remove_stopwords(list_tokens=tokenizer(x)))
    return aux


def translate_description_series(
    dataframe: pd.DataFrame, column_name: str, target_lang: str = 'pt'
) -> pd.Series:
    """Translate columns to another language.

    Args:
        dataframe (pd.DataFrame): dataframe with column to be translated
        column_name (str): column name to be translated
        target_lang (str): taget language

    Returns:
        pd.Series: column translated
    """
    dataframe[column_name] = dataframe[column_name].fillna("")
    dataframe[column_name] = dataframe[column_name].swifter.apply(
        lambda x: translate_description(x, target_lang)
        if isinstance(x, str) else x
    )
    return dataframe[column_name]


def translate_description(text: str, target_lang: str = 'pt') -> str:
    """Translate non-portuguese text.

    Args:
        text (str): column name to be translated
        target_lang (str): taget language

    Returns:
        str: text translated
    """
    try:
        return GoogleTranslator(
            source='auto', target=target_lang).translate(text)
    except Exception:
        return text
