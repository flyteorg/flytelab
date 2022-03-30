import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


class TextVectorizer():
  
  def __init__(self,model:str='neuralmind/bert-base-portuguese-cased'):
    self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False)

    self.model = AutoModel.from_pretrained(model)
  
  def encode_inputs(self,series_text):

    input_ids = self.tokenizer(list(series_text), padding=True, 
                                            truncation=True, 
                                            max_length=512,
                                            return_tensors="pt",
                                            add_special_tokens=True)                                            
    return input_ids


  def get_df_embedding(self,input_ids:pd.Series)-> pd.Series:
          """
          Method responsable for generate pd.DataFrame with all text vector representation
          Args:
              model_path
              input_ids (torch.DataLoader): DataLoader responsable for generate text batches
          Return:
              dataframe with all text converted to text vector representation
          """
          with torch.no_grad():
            outs = self.model(input_ids['input_ids'])[0][:,1:-1,:].mean(axis=1).numpy()
          return pd.DataFrame(outs)