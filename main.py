import requests
from bs4 import BeautifulSoup
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

openai.api_key = "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"  # os.getenv("OPENAI_API_KEY")


## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
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

# url = ("https://digitelmobile.tel-aviv.gov.il/SharepointData/api/ListData/DigitelNews/%D7%90%D7%A4%D7%9C%D7%99%D7%A7"
#        "%D7%A6%D7%99%D7%99%D7%AA%20106")
url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679&from=EN"
df = pd.DataFrame({
    "title": [],
    "heading": [],
    "content": [],
    "tokens": [],
    "url": []
})

try:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    legislation = soup.text

    # Now we split the legislation into sections, each section starts with the world "Artikel"
    search_document = legislation.split("HAVE ADOPTED THIS REGULATION:")
    sections = search_document[1].split('\nArticle')
    sections = ["Article" + section for section in sections]
    sections = sections[1:]

    print(f'Amount of Articles: {len(sections)}')

    section_titles = soup.find_all(class_='sti-art')
    section_titles = [title.text for title in section_titles]

    section_titles[:5]

    # We can now parse each section using tiktoken, and calculate the amount of tokens per section
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens_per_section = []

    for section in sections:
        tokens = enc.encode(section)
        tokens_per_section.append(len(tokens))

    # Create a loop of 99 iterations
    headings = []
    for i in range(99):
        headings.append("Article " + str(i + 1))

    df['title'] = section_titles
    df['heading'] = headings
    df['content'] = sections
    df['tokens'] = tokens_per_section

    df = df.set_index(["title", "heading"])
    print(f"{len(df)} rows in the data.")

    document_embeddings = compute_doc_embeddings(df)

    # An example embedding:
    example_entry = list(document_embeddings.items())[0]
    print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

    items = order_by_similarity("Can the commission implement acts for exchanging information?", document_embeddings)[:5]
    items = order_by_similarity("Am I allowed to delete my personal information?", document_embeddings)[:5]
    print(items)

    # content = json.loads(response.content)
    #
    # for item in content:
    #     df1 = pd.DataFrame({
    #         "content": [item['announText']],
    #         "url": [item['id']]
    #     })
    #     df = pd.concat([df, df1])
    #
    #     df1 = pd.DataFrame({
    #         "heading": [item['brief']],
    #         "url": [item['id']]
    #     })
    #     df = pd.concat([df, df1])

except requests.exceptions.HTTPError as error:
    print(error)
