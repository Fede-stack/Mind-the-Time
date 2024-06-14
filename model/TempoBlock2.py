import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Embedding
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

def set_trainable_bias_only(self, encoder):
        """
        Sets only the bias terms of the encoder layers to be trainable.
        """
        transformer_model = getattr(encoder, 'transformer', encoder)
        for layer in transformer_model.layers:
            layer.trainable = False  # Set the entire layer as non-trainable
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    if 'bias' in weight.name:
                        weight._handle_name = weight.name  # Set handle name for weight
                        encoder.add_weight(
                            name=weight.name,
                            shape=weight.shape,
                            trainable=True,  # Make bias trainable
                            initializer=tf.constant_initializer(weight.numpy())
                        )

class ProjectionLayer(layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(ProjectionLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def build(self, input_shape):
        # create layer weights, shape = (input_dim, output_dim)
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='uniform',
            trainable=True,
            name='embeddings'
        )
    
    def call(self, inputs):
        # convert inputs to integers
        inputs = tf.cast(inputs, tf.int32)
        # lookup embeddings for input indices
        return tf.nn.embedding_lookup(self.embeddings, inputs)
    
    def get_config(self):
        # enable layer serialization
        config = super(ProjectionLayer, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
        })
        return config

# define input layers
input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32)
attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32)
token_type_ids = Input(shape=(MAX_LEN,), dtype=tf.int32)

# load pretrained model
encoder = TFAutoModel.from_pretrained(MODEL_NAME, return_dict=True, output_hidden_states=True)
encoder.resize_token_embeddings(len(tokenizer))

# get encoder outputs
outputs = encoder({"input_ids": input_ids, "attention_mask": attention_mask}, training=True)
pooler_outputs = outputs.last_hidden_state

# input embeddings
inputs = Input(shape=(1,))
emb1 = layers.Embedding(len(np.unique(year)), 2)
outs_1 = emb1(inputs)

#project embeddings into EMBED_DIM
xt = layers.Dense(EMBED_DIM, activation='linear')(outs_1)
xt = layers.Reshape((1, EMBED_DIM))(xt)
memento = layers.Attention()
hier = []

#hierarchical attention mechanism
for i in range(MAX_LEN):
    resh = layers.Reshape((1, EMBED_DIM))(pooler_outputs[:, i, :])
    conc = layers.Concatenate(axis=1)([resh, xt])
    att_t = memento([conc, conc, conc])
    att_t = tf.reduce_sum(att_t, axis=1)
    att_t = layers.Reshape((1, EMBED_DIM))(att_t)
    hier.append(att_t)

# concatenating hierarchical outputs
summed_output = layers.Concatenate(axis=1)(hier)
attention_output = layers.Dropout(0.1, name="encoder_2/att_dropout")(summed_output)
attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

# feed-forward layer
ffn = tf.keras.Sequential([
    layers.Dense(128, activation="gelu"),
    layers.Dense(EMBED_DIM),
])

ffn_output = ffn(attention_output)
ffn_output = layers.Dropout(0.1)(ffn_output)

#adding residual connection
sequence_output = layers.LayerNormalization(epsilon=1e-6)(pooler_outputs + ffn_output)

#global average pooling
outs = layers.GlobalAveragePooling1D()(sequence_output[:, 1:, :])

# output layer with softmax activation
output = layers.Dense(vocab_size, activation="softmax")(sequence_output)

#defining the model
model = Model(inputs=[[input_ids, attention_mask], inputs], outputs=[output])
opt = tf.keras.optimizers.Adam(learning_rate=1e-7)

#compiling the model with mixed precision loss scaling
model.compile(optimizer=tf.keras.mixed_precision.LossScaleOptimizer(opt), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), 
              metrics=["accuracy"])

tf.random.set_seed(0)
history = model.fit(
    [x_train, embs_train_year],
    y_train.astype(np.int32),
    epochs=2,
    shuffle = True,
    batch_size=12,
    sample_weight=sample_weights
    )
