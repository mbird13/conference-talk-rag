import sys
import os
from openai import OpenAI
import json

def read_questions(file_path):
    """Read questions from a file, one per line."""
    with open(file_path, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

def read_talks(file_path):
    """Read talks from a JSON file."""
    with open(file_path, 'r') as f:
        talks = json.load(f)
    return talks

def generate_answers(questions, talks):
    """Use OpenAI API to answer questions based on talks."""
    with open("config.json") as config:
        openaiKey = json.load(config)["openaiKey"]
    OpenAI.api_key = openaiKey
    client = OpenAI(api_key=OpenAI.api_key)
    
    answers = []
    for i, question in enumerate(questions):
        talks_context = "\n\n".join([
            f"Title: {talk['talk']['title']}\nSpeaker: {talk['talk']['speaker']}\n{talk['talk']['text']}"
            for talk in talks if talk['question_idx'] == i
        ])
        
        prompt = f"""Based on the following talks, answer this question: {question}

Talks:
{talks_context}

Provide a thoughtful answer grounded in the talks provided."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        answers.append({
            "question": question,
            "answer": response.choices[0].message.content
        })
    
    return answers


if len(sys.argv) != 3:
    print("provide a question file and a similarity file")
    sys.exit(1)

questions_file = sys.argv[1]
talks_file = sys.argv[2]

questions = read_questions(questions_file)
talks = read_talks(talks_file)

answers = generate_answers(questions, talks)

for result in answers:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}\n")

