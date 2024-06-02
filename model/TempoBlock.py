import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Embedding
from transformers import TFAutoModel, AutoTokenizer
import numpy as np

class TempoBlockModel:
    def __init__(self, model_name, max_len, embed_dim, vocab_size, tokenizer):
        self.MODEL_NAME = model_name
        self.MAX_LEN = max_len
        self.EMBED_DIM = embed_dim
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.model = self.build_model()

    def build_model(self):
        # Define input layers
        input_ids = Input(shape=(self.MAX_LEN,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.MAX_LEN,), dtype=tf.int32, name='attention_mask')
        token_type_ids = Input(shape=(self.MAX_LEN,), dtype=tf.int32, name='token_type_ids')
        
        # Load pre-trained encoder
        encoder = TFAutoModel.from_pretrained(self.MODEL_NAME, return_dict=True, output_hidden_states=True, from_pt=True)
        encoder.resize_token_embeddings(len(self.tokenizer))
        
        # Get the encoder outputs
        encoder_outputs = encoder({"input_ids": input_ids, "attention_mask": attention_mask}, training=True)
        pooler_outputs = encoder_outputs.last_hidden_state

        # Additional input for temporal embeddings 
        time_input = Input(shape=(1,))
        emb1 = Embedding(len(np.unique(year)), 2)(time_input)

        # Project Embeddings into EMBED_DIM
        xt = Dense(self.EMBED_DIM, activation='linear')(emb1)
        xt = layers.Reshape((1, self.EMBED_DIM))(xt)

        # Hierarchical Attention mechanism
        hier = []
        for i in range(self.MAX_LEN):
            resh = layers.Reshape((1, self.EMBED_DIM))(pooler_outputs[:, i, :])
            conc = layers.Concatenate(axis=1)([resh, xt])
          
            mha2 = layers.Attention()([conc, conc, conc])
            mha2 = tf.reduce_sum(mha2, axis=1)
          
            mha2 = layers.Reshape((1, self.EMBED_DIM))(mha2)
            hier.append(mha2)

        # Combine hierarchical outputs
        summed_output = layers.Concatenate(axis=1)(hier)
        attention_output = layers.Dropout(0.1, name="encoder_attention_dropout")(summed_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

        # Feed-forward layer
        ffn = tf.keras.Sequential([
            layers.Dense(128, activation="gelu"),
            layers.Dense(self.EMBED_DIM),
        ])
        ffn_output = ffn(attention_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)

        # Combine feed-forward output with encoder outputs
        sequence_output = layers.LayerNormalization(epsilon=1e-6)(pooler_outputs + ffn_output)

        # Global Average Pooling
        outs = layers.GlobalAveragePooling1D()(sequence_output[:, 1:, :])
        output = Dense(self.vocab_size, activation="softmax")(sequence_output)

        # Define the model
        model = Model(inputs=[input_ids, attention_mask, time_input], outputs=output)

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=1e-7)
        model.compile(optimizer=tf.keras.mixed_precision.LossScaleOptimizer(opt), 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), 
                      metrics=["accuracy"])
        
        return model

# model = TempoBlockModel(MODEL_NAME, MAX_LEN, EMBED_DIM, vocab_size, tokenizer)
