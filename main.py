import os
import requests
import json
import numpy as np
import pandas as pd
import openai
import tiktoken

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-4"
MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003  

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

openai.api_key = "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" #os.getenv("OPENAI_API_KEY")

## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    print(df)
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
        
    return chosen_sections, chosen_sections_len

def answer_with_gpt_4(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    messages = [
        {"role" : "system", "content":"You are a GDPR chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'"}
    ]
    prompt, section_lenght = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)

    context= ""
    for article in prompt:
        context = context + article 

    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})
    response = openai.ChatCompletion.create(
        model=COMPLETIONS_MODEL,
        messages=messages
        )

    return '\n' + response['choices'][0]['message']['content'], section_lenght

if __name__ == '__main__':

    url = "https://digitelmobile.tel-aviv.gov.il/SharepointData/api/ListData/DigitelNews/%D7%90%D7%A4%D7%9C%D7%99%D7%A7%D7%A6%D7%99%D7%99%D7%AA%20106"

    df = pd.DataFrame({
        "title": [],
        "heading":[],
        "content":[],
        "tokens":[],
        "url": []
    })

    try:
        response = requests.get(url)
        content = json.loads(response.content)

        for item in content:
            df1 = pd.DataFrame({
                "content": [item['announText']],
                "url": [item['id']]
            })
            df = pd.concat([df, df1])

            df1 = pd.DataFrame({
                "heading": [item['brief']],
                "url": [item['id']]
            })
            df = pd.concat([df, df1])

    except requests.exceptions.HTTPError as error:
        print(error)     

    document_embeddings = compute_doc_embeddings(df)           

    # An example embedding:
    example_entry = list(document_embeddings.items())[0]
    print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

    prompt = "תאריך הבחירות"
    response, sections_tokens = answer_with_gpt_4(prompt, df, document_embeddings)
    print(response)    
