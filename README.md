# HYPOGEN: ADVERSARIAL HYPOTHESIS GENERATION AND STRUCTURED REPRESENTATION FROM SCIENTIFIC LITERATURE
Generating scientific hypotheses with LLMs.

## Pipeline

The pipeline consists of the following scripts which are largely run in sequential order.

## `preprocessing.py`
`preprocessing.py` serves as the initial pipeline for hypothesis extraction from academic abstracts based on the Bit-Flip concept. Utilising a custom prompt and a fine-tuned language model hosted on Azure, the script takes in research abstracts from ArXiv as input and outputs structured representations of the Bit and Flip—the prevailing belief being challenged and the counterargument, respectively. This data is captured in a JSON-style dictionary and further serialised for easy storage and subsequent analysis. The script is designed to be scalable and employs multi-threading to expedite the processing of large datasets. This enables it to handle batches of abstracts, either by appending to an existing dataset or creating a new one.

## `embedding.py`
`embedding.py` is responsible for converting the research abstracts into vectorised embeddings. The script uses a pre-trained language model to transform textual data into a continuous vector space. By utilising OpenAI's language embedding services, it generates embeddings in chunks for enhanced performance. It interfaces with a CSVLoader to pull in the abstract data, typically stored in a CSV file, and uses it as the input for the embedding process. The generated embeddings are stored in a Chroma database, which provides a robust persistence layer, making it easy to access and manipulate the embeddings at later stages of the pipeline. This module serves as a bridge between the initial hypothesis extraction and the subsequent machine learning tasks, facilitating easier similarity searches and comparisons among research topics.
