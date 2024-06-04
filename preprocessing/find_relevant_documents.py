def find_relevant_documents(documents, keywords):
  """Finds all documents containing AT LEAST ONE of the keywords.

  Args:
      documents: A list of strings representing the documents.
      keywords: A list of strings representing the keywords to search for.

  Returns:
      A tuple containing a list of relevant documents and a list of their indices in the original list.
  """
    relevant_docs = []
    idx_ = []
    for i, doc in enumerate(documents):
        if any(word in doc for word in keywords):
            relevant_docs.append(doc)
            idx_.append(i)
    return relevant_docs, idx_

# rel_documents, idx_ = find_relevant_documents(documents, words_to_consider)
