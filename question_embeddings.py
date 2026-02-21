import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import os
import shutil
import json
from openai import OpenAI

import sys

def free_embeddings(lines):
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Using GPU for encoding")
    else:
        print("Using CPU for encoding")

    #get embeddings of each line of the input file
    embeddings = model.encode(
        lines,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    return embeddings

def openai_embeddings(lines):
    with open("config.json") as config:
        openaiKey = json.load(config)["openaiKey"]
    
    OpenAI.api_key = openaiKey
    client = OpenAI(api_key=OpenAI.api_key)
    # Process current batch
    response = client.embeddings.create(input=lines, model='text-embedding-3-small')
    embeddings = [item.embedding for item in response.data]
    return embeddings

if __name__ == "__main__":
    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "questions"
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    input_file = sys.argv[1]
    if not input_file:
        print("No input file provided")
        sys.exit(1)

    with open(input_file, 'r') as file:
        lines = file.readlines()
    lines = [line.replace("\n", " ") for line in lines]

    free_df = pd.DataFrame(lines, columns=['questions'])
    open_df = pd.DataFrame(lines, columns=['questions'])
    
    free_df['embedding'] = free_embeddings(lines)
    open_df['embedding'] = open_embeddings = openai_embeddings(lines)

    output_file = os.path.join(output_dir, "free_questions.csv")
    free_df.to_csv(output_file, index=False)

    output_file = os.path.join(output_dir, "openai_questions.csv")
    open_df.to_csv(output_file, index=False)









