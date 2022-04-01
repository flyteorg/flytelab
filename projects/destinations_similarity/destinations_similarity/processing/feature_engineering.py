"""Feature engineering for the data."""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


class TextVectorizer():
    """Class used to vectorize text."""

    def __init__(self, model: str = 'neuralmind/bert-base-portuguese-cased') -> None:
        """Initialize class.
    
        Args:
            model (str): huggingface path    
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, do_lower_case=False)
        self.model = AutoModel.from_pretrained(model)

    def encode_inputs(self, series_text: pd.Series) -> torch.tensor:
        """Encode inputs.
    
        Args:
            series_text (pd.Series): text to be vectorized
            
        Returns:
            torch.tensor: tokens ids
        """
        input_ids = self.tokenizer(
            list(series_text), padding=True, truncation=True,
            max_length=256, return_tensors="pt", add_special_tokens=True
        )
        return input_ids

    def get_df_embedding(self, input_ids: pd.Series) -> pd.DataFrame:
        """Generate DataFrame with all text vector representations.
    
        Args:
            input_ids (torch.tensor): tokens ids
            
        Returns:
            pd.DataFrame: input id vectors
        """
        with torch.no_grad():
            outs = self.model(
                input_ids['input_ids']
            )[0][:, 1:-1, :].mean(axis=1).numpy()
        return pd.DataFrame(outs)
