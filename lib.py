import pandas as pd
import os 
import ollama
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import tqdm
# UTILITY FUNCTIONS 
def clean_text(text):
   text = re.sub(r'\W', ' ', str(text))
   text = re.sub(r'\s+', ' ', text)
   text = text.lower()
   return text

def process_description(description):
   cleaned_text = clean_text(description)
   return cleaned_text

def prepare_data(df):
    """
    Helper function to prepare data and return a new DataFrame with 'text' and 'embedding' columns.
    :param df: Input pandas DataFrame.
    :param get_embeddings: Function to generate embeddings for a given text.
    :return: DataFrame with two columns: 'text' and 'embedding'.
    """
    # Create an empty list to store the text and embeddings
    data = []

    # Iterate through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        # Convert the row to a dictionary where the column names are keys
        row_dict = {col: str(row[col]) for col in df.columns}        
        # Flatten the dictionary into a string where each key-value pair is joined by a separator
        flat_text = ', '.join([f"{key}: {value.replace('nan', '')}" for key, value in row_dict.items()])
        # Generate the embedding using the provided function
        embedding = ollama.embed(model='wizardlm2', input=[flat_text])
        # Append the flat text and embedding as a tuple to the data list
        data.append({'nc': flat_text, 'vector': embedding})
    # Convert the list of dictionaries into a DataFrame
    result_df = pd.DataFrame(data, columns=['nc', 'vector'])
    
    return result_df