import gc
import random

def find_neural_network_indices(word_list, parola):
    for i in range(len(word_list) - 1):
        if word_list[i] == parola :#and word_list[i + 1] == 'learning':
            if i < MAX_LEN:
                return i
    return None

def plot_linear_regression(x, y):
    # Convert x and y to numpy arrays if they are not already
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    
    # Compute the linear regression
    model = LinearRegression(fit_intercept=False).fit(x, y)
    y_pred = model.predict(x)
    
    # Compute Pearson and Spearman correlation coefficients
    pearson_coef, _ = stats.pearsonr(x.flatten(), y)
    spearman_coef, _ = stats.spearmanr(x.flatten(), y)
    
    # Plot the data and the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Original data')
    plt.plot(x, y_pred, color='red', label='Regression line')
    plt.title(f'Linear Regression (Pearson: {pearson_coef:.2f}, Spearman: {spearman_coef:.2f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

ids = []
masks = []
embs_test_year = []
for i in range(len(raw)):
  inputs = tokenizer(raw[i], max_length=MAX_LEN, padding='max_length', truncation=True, return_attention_mask=True, return_token_type_ids=True, add_special_tokens=True)
  ids.append(inputs['input_ids'])
  masks.append(inputs['attention_mask'])
ids = np.array(ids)

x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels( ids )
training_set = prepare_dataset(x_masked_train, y_masked_labels, masks)

emb_word_extract = Model(inputs=model.inputs, outputs=sequence_output)

coss = []
nns = []
np.random.seed(20)
random.seed(20)
for parola in ann_list:
    nn = []
    for ii in np.unique(year):
        if ii == 0:
            docs_t = data1
        else:
            docs_t = data2
        
        # Filtra i documenti che contengono la parola
        docs_t = [doc for doc in docs_t if parola in doc.lower()]
        
        # Se non ci sono documenti, continua con l'iterazione successiva
        if not docs_t:
            nn.append(np.repeat(0.001, 768))
            continue

        # Riduci la dimensione del campione se necessario
        n_samples = min(len(docs_t), 1000)
        docs_t = random.sample(docs_t, n_samples)

        # Pre-elaborazione e tokenizzazione dei documenti
        ids, masks, idx_ = [], [], []
        for doc in docs_t:
            inputs = tokenizer(doc, max_length=MAX_LEN, padding='max_length', truncation=True, return_attention_mask=True, return_token_type_ids=True, add_special_tokens=True)
            ids.append(inputs['input_ids'])
            masks.append(inputs['attention_mask'])
            idx_.append(find_neural_network_indices(tokenizer.convert_ids_to_tokens(tokenizer.encode(doc)), parola))

        # Creazione del training set
        if ids and masks:
            x_masked_train, y_masked_labels, _ = get_masked_input_and_labels(np.array(ids))
            training_set = {'input_ids': x_masked_train, 'attention_mask': np.array(masks), 'labels': y_masked_labels}
            input_test = [np.array(training_set['labels'], dtype=int), np.array(training_set['attention_mask'], dtype=int)]
            tf.random.set_seed(0)
            
            # Prova a estrarre gli embeddings
            try:
                embs_nn = emb_word_extract.predict([input_test, np.repeat(ii, n_samples)])
                embs = [embs_nn[i, idx, :] for i, idx in enumerate(idx_) if idx is not None and idx < MAX_LEN]
                nn.append(np.mean(embs, axis=0) if embs else np.repeat(0, 768))
            except ValueError as e:
                print(f"Skipping due to error: {e}")
                nn.append(np.repeat(0, 768))
        else:
            print("Skipping prediction due to empty inputs.")
            nn.append(np.repeat(0, 768))
        
        # Pulizia della memoria
        del ids, masks, x_masked_train, y_masked_labels, training_set, input_test, embs_nn
        gc.collect()

    # Calcolo della similarità del coseno
    coss.append(1 - cosine_similarity([nn[0], nn[1]])[0][1])
    nns.append(nn)
