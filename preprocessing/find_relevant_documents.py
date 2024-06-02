def find_relevant_documents(documents, keywords):
  """
  find the relevant documents based on keywords.
  keywords is a list containing the words to consider for LSC task (e.g., for the English dataset there are 37 words)
  """
    relevant_docs = []
    idx_ = []
    for i, doc in enumerate(documents):
        if any(word in doc for word in keywords):
            relevant_docs.append(doc)
            idx_.append(i)
    return relevant_docs, idx_

# rel_documents, idx_ = find_relevant_documents(documents, words_to_consider)
