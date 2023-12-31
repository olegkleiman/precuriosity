#
# Created by Erez Roted, supervised by Loren Taboo
#

import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import numpy as np
import tiktoken
import openai

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-4"
MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

openai.api_key = "__sk-uk0I6v0yTdajwETf2dZAT3BlbkFJmLY5CQ3hJGMmi7dUEotx__"  # os.getenv("OPENAI_API_KEY")

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    embeddings = result["data"][0]["embedding"]
    return embeddings


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


def remove_tags(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script', 'p']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


df = pd.DataFrame({
    "title": [],
    "heading": [],
    "content": [],
    "tokens": [],
    "url": [],
    "id": [],
})

try:
    enc = tiktoken.encoding_for_model("gpt-4")
    cnxn_str = ("Driver={SQL Server Native Client 11.0};"
                "Server=tcp:tlvsearch.database.windows.net,1433;"
                "Database=curiosity;"
                "UID=okey;"
                "PWD=dfnc94^*;")

#"Server=tcp:tlvsearch.database.windows.net,1433;Initial Catalog=curiosity;Persist Security Info=False;User ID=okey;Password={your_password};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"

    cnxn = pyodbc.connect(cnxn_str)
    queryDf = pd.read_sql("select * from [dbo].[config] where is_enabled = 1", cnxn)
    cnxn.close()

    # data = {
    #     'url': [
    #         "https://digitelmobile.tel-aviv.gov.il/SharepointData/api/ListData/אירועים/mobileapp",
    #         # "https://digitelmobile.tel-aviv.gov.il/SharepointData/api/ListData/כתבות/mobileapp",
    #         "https://digitelmobile.tel-aviv.gov.il/SharepointData/api/ListData/הודעות דיגיתל/mobileapp",
    #         "https://digitelmobile.tel-aviv.gov.il/SharepointData/api/ListData/מבזקים/מבזקים בתוקף",
    #         "https://digitelmobile.tel-aviv.gov.il/SharepointData/api/ListData/פרוייקטים תכנון עיר/mobileapp",
    #         "https://digitelmobile.tel-aviv.gov.il/SharepointData/api/ListData/הטבות/mobileapp"
    #     ],
    #     'title_map': [
    #         "title",
    #         # "title",
    #         "title",
    #         "title",
    #         "title",
    #         "title"],
    #     "heading_map": [
    #         "summary",
    #         # "summary",
    #         "summary",
    #         "summary",
    #         "summary",
    #         "title"
    #     ],
    #     "content_map": [
    #         "comments",
    #         # "content",
    #         "details",
    #         "content",
    #         "summary",
    #         "remarks"],
    #     "url_map": [
    #         "previewPage",
    #         # "_x05ea__x05e6__x05d5__x05d2__x05",
    #         "fileRef",
    #         "fileRef",
    #         "fileRef",
    #         "previewPage"],
    #     "id_map": [
    #         "id",
    #         # "id",
    #         "id",
    #         "id",
    #         "id",
    #         "id"]
    # }
    # queryDf = pd.DataFrame(data)

    for index, ref in queryDf.iterrows():
        url = ref["url"]
        response = requests.get(url)
        content = json.loads(response.content)

        title_column_name = ref["title_map"]
        content_column_name = ref["content_map"]
        head_column_name = ref["heading_map"]
        url_column_name = ref["link_map"]
        id_column_name = ref["id_map"]

        for item in content:
            doc = item[content_column_name]
            if doc is None:
                continue

            clean_doc = remove_tags(doc)

            tokens = enc.encode(clean_doc)

            if len(tokens) > 0:
                df1 = pd.DataFrame({
                    "title": [item[title_column_name]],
                    "content": clean_doc,
                    "heading": [item[head_column_name]],
                    "url": [item[url_column_name]],
                    "tokens": [len(tokens)],
                    "id": [item[id_column_name]]
                })
                df = pd.concat([df, df1], ignore_index=True)

    document_embeddings = compute_doc_embeddings(df)

    docs = order_by_similarity(  # "חצר טבעית",
        "שנת לימודים",
        document_embeddings)
    docs = docs[:5]
    print(docs)

    docIndex = docs[0][1]
    print(df.iloc[docIndex])

except requests.exceptions.HTTPError as error:
    print(error)
