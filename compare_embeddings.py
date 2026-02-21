import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Load data
free_talks = pd.read_csv('free/free_talks.csv')
free_paragraphs = pd.read_csv('free/free_paragraphs.csv')
free_clusters = pd.read_csv('free/free_3_clusters.csv')
free_questions = pd.read_csv('questions/free_questions.csv')

openai_talks = pd.read_csv('openai/openai_talks.csv')
openai_paragraphs = pd.read_csv('openai/openai_paragraphs.csv')
openai_clusters = pd.read_csv('openai/openai_3_clusters.csv')
openai_questions = pd.read_csv('questions/openai_questions.csv')


def find_top_3_similar(question_embeddings, target_embeddings, target_df, target_name):
    """Find top 3 most similar embeddings"""
    similarities = cosine_similarity(question_embeddings, target_embeddings)
    top_indices = np.argsort(similarities, axis=1)[:, -3:][:, ::-1]
    
    results = []
    for question_index, indices in enumerate(top_indices):
        for rank, talk_index in enumerate(indices, 1):
            results.append({
                'question_idx': question_index,
                'rank': rank,
                target_name: target_df.iloc[talk_index].to_dict()
            })
    return results

def process_embeddings(data_type, talks_df, paragraphs_df, clusters_df, questions_df):
    """Process all embeddings for a data type"""
    
    question_embs = np.array([eval(e) for e in questions_df['embedding']])
    
    talk_results = find_top_3_similar(question_embs, np.array([eval(e) for e in talks_df['embedding']]), talks_df, 'talk')
    para_results = find_top_3_similar(question_embs, np.array([eval(e) for e in paragraphs_df['embedding']]), paragraphs_df, 'paragraph')
    cluster_results = find_top_3_similar(question_embs, np.array([eval(e) for e in clusters_df['embedding']]), clusters_df, 'cluster')
    
    # Write results to files
    with open(f'{data_type}_talks_similarities.txt', 'w') as f:
        f.write(json.dumps(talk_results, indent=2))
    with open(f'{data_type}_paragraphs_similarities.txt', 'w') as f:
        f.write(json.dumps(para_results, indent=2))
    with open(f'{data_type}_clusters_similarities.txt', 'w') as f:
        f.write(json.dumps(cluster_results, indent=2))

# Process both embedding types
process_embeddings('free', free_talks, free_paragraphs, free_clusters, free_questions)
process_embeddings('openai', openai_talks, openai_paragraphs, openai_clusters, openai_questions)

print("Similarity analysis complete. Results saved to text files.")