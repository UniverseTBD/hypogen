# HYPOGEN: ADVERSARIAL HYPOTHESIS GENERATION AND STRUCTURED REPRESENTATION FROM SCIENTIFIC LITERATURE
Generating scientific hypotheses with LLMs.

## `preprocessing.py`
`preprocessing.py` serves as the initial pipeline for hypothesis extraction from academic abstracts based on the Bit-Flip concept. Utilising a custom prompt and a fine-tuned language model hosted on Azure, the script takes in research abstracts from ArXiv as input and outputs structured representations of the Bit and Flip—the prevailing belief being challenged and the counterargument, respectively. This data is captured in a JSON-style dictionary and further serialised for easy storage and subsequent analysis. The script is designed to be scalable and employs multi-threading to expedite the processing of large datasets. This enables it to handle batches of abstracts, either by appending to an existing dataset or creating a new one.
