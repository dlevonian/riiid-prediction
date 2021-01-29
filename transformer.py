# Transformer Class and its components
# generally following the architecture of Attention is all you need, Vaswani et al, 2017
# - positional encoding
# - masking: only look-ahead, upper-triangular mask
# - atomic QKV attention
# - Multi-Head Attention (MHA)
# - Encoder Layer
# - Decoder Layer
# - Encoder, Decoder Classes

# ============== DATAFRAME
N_Q = 13523
REMOVE_LECTURES=True
LAG_BINS = [-1e8,0,0.1,5,10,15,20,25,30,35,40,50,60,70,80,90,100,125,150,200,300,500,3600,36000]
N_LAGBINS = len(LAG_BINS)
HIST_BINS = [1,13,28,51,83,124,173,232,302,385,484,604,749,929,1155,1446,1832,2382,3251,4945]
N_HISTBINS = len(HIST_BINS)

# ============== DATASET
WINDOW = 160
STEP = 80
BATCH_SIZE = 128
D_TYPE = tf.float32

# ============== TRANSFORMER
D_MODEL = 256
D_POS = 16
N_LAYERS = 2
N_HEADS=8
DFF=D_MODEL
DROPOUT=0.1



def positional_encoding(seq_len=WINDOW, d_pos=D_POS, min_rate=1e-2):
    """return sin/cos position indicator  
       in contrast to AIAYN, the positional encoding is concatented, not added
       no need to use full D_MODEL, a separate smaller dimension is used (e.g. 16 or 32) 
    """
    angle_rates = min_rate**(np.linspace(0,1,int(d_pos)))
    # mesh position*min_rate**(d/d_pos)
    angles = np.arange(seq_len)[:, np.newaxis] * angle_rates[np.newaxis, :]  
    return tf.cast(np.sin(angles), dtype=D_TYPE)


def decoder_mask(seq):
    """produce the decoder mask for the MHA (padding mask + look-ahead mask)
    a function rather that constant b/c padding is different for different input seqs   
    Input:  (batch_size, seq_len)
    Output: (batch_size, num_heads, seq_len, seq_len)
    """
    # insert 2 dimensions into padding mask (num_heads, seq_k)
    padding_mask = tf.cast(tf.math.equal(seq, 0), D_TYPE)[:, tf.newaxis, tf.newaxis, :]
    
    # look_ahead (upper triangular mask)
    seq_len = tf.shape(seq)[-1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=D_TYPE), -1, 0)

    return tf.maximum(padding_mask, look_ahead_mask)


def QKV_attention(Q, K, V, mask):
    """
    Args:
        Q: (..., seq_len_q, d_q)  
              - depth (d_q) can be d_model or d_model/H for MHA
              - seq_len_q can be same as seq_len_k (self-attention) or different (Enc-Dec attention)
        K: (..., seq_len_k, d_k)     # d_k=d_q
              - must have same depth as Q  b/c depth is convolved in QK dot product to produce attn.weights
        V: (..., seq_len_v, d_v)     # seq_len_v=seq_len_k
              - d_v can be arbitary but in practice is used the same as d_q=d_k
              - seq_len_v=seq_len_k for proper key-value mapping
              
        mask: 1/0 tf.float32 tensor broadcastable to attention_weights shape (..., seq_len_q, seq_len_k)  
    Returns:
      attn_output   (..., seq_len_q, d_v)  
    """
    # 1. ATTENTION QUERY - (..., seq_len_q, seq_len_k) - which queries should 'attend' to which keys
    QK = tf.matmul(Q, K, transpose_b=True) 
  
    # 2. SCALE by 1/std.dev=sqrt(d_k) - to neutralize the effect of long vs. short embeddings
    scaled_QK = QK / tf.math.sqrt( tf.cast(tf.shape(K)[-1], dtype=D_TYPE) )

    # 3. MASK - add to scaled_QK to make padded zeros large negative numbers (arguments to softmax)
    #    K is masked (not Q nor V) to prevent quering Keys from (1) padded or (2) future tokens
    scaled_QK = scaled_QK + mask*-1e9

    # 4. SOFTMAX across K - sharpen the attention at the most relevant keys
    attn_weights = tf.nn.softmax(scaled_QK, axis=-1)
    
    # 5. Softmaxed keys select relevant VALUES
    attn_output = tf.matmul(attn_weights, V)
    return attn_output


class MHA(tf.keras.layers.Layer):
    """
    MHA is a layer used both in Encoder (self-attn) and Decoder (self-attn & Enc-Dec attn)
    MHA splits d_model into H heads which scan the sequence independently, then concatenate. 
    Different head can attend to different features at different positions 
    """
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model%n_heads==0

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head  = tf.cast(d_model/n_heads, tf.int32)  

        # Linear fully-coonected projection layers Z->QKV
        self.linear_q = Dense(d_model)
        self.linear_k = Dense(d_model)
        self.linear_v = Dense(d_model)
        # Final linear layer that links concatenated heads to the d_model-dimensional output
        self.linear_output = Dense(d_model)
        
    def call(self, Q, K, V, mask):
        batch_size =  tf.shape(Q)[0]

        Q = self.linear_q(Q)  # (batch_size, seq_len_q, d_model)
        K = self.linear_k(K)  # (batch_size, seq_len, d_model)
        V = self.linear_v(V)  # (batch_size, seq_len, d_model)
        
        # Split d_model into heads:  (batch_size, seq_len_q, num_heads, d_head)
        Q = tf.reshape(Q, (batch_size, -1, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch_size, -1, self.n_heads, self.d_head))
        V = tf.reshape(V, (batch_size, -1, self.n_heads, self.d_head))

        # QKV_attention needs seq_len in the -2 position: (batch_size, num_heads, seq_len, d_head)
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        # apply QKV_attention to each head. (batch_size, num_heads, seq_len_q, d_head)
        output = QKV_attention(Q, K, V, mask)

        # Bring num_heads into penultimate position and concatenate back to get num_heads * d_head = d_model
        output = tf.transpose(output, perm=[0, 2, 1, 3])            # (batch_size, seq_len_q, num_heads, d_head)
        output = tf.reshape(output, (batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)

        output = self.linear_output(output)  # (batch_size, seq_len_q, d_model)
            
        return output    


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dff, dropout_rate): 
        super().__init__()

        self.mha = MHA(d_model, n_heads)
        self.dense_relu = Dense(dff, activation='relu')
        self.linear = Dense(d_model)
        
        # dropout to be implemented before adding residual and normalization
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        # Layer Norm. -- normalize activations across features (d_model) within each x
        # A functional transpose of the Batch Norm, which normalizes each feature across all x in a batch
        # By deafault, applied to last dimension of the tensor. Epsilon is added to the variance to avoid dividing by 0.
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
    
    def call(self, x, mask, training):

        # Encoder self-attention uses the same input x for Q, K, V
        # All steps produce the same output shape  (batch_size, seq_len, d_model)
        self_attn = self.mha(x, x, x, mask)

        # Dropout training: True in training mode (w/ dropout), False in inference mode (no dropout)
        self_attn = self.dropout1(self_attn, training=training) 
        self_attn = self.layernorm1(self_attn + x)  # add residual connection and normalize
        
        # feed-forward part - first expanding to DFF, then brining back to D_MODEL
        output = self.dense_relu(self_attn)
        output = self.linear(output)

        output = self.dropout2(output, training=training) 
        output = self.layernorm2(output + self_attn)  # add residual connection and normalize
        
        return output  # shape=(batch_size, seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dff, dropout_rate):
        super().__init__()

        self.mha1 = MHA(d_model, n_heads)
        self.mha2 = MHA(d_model, n_heads)

        self.dense_relu = Dense(dff, activation='relu')
        self.linear = Dense(d_model)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        
    def call(self, x, enc_output, mask, training):
        # All steps produce the same output shape  (batch_size, target_seq_len, d_model)
        # cross_attn uses enc_output (batch_size, input_seq_len, d_model)

        self_attn = self.mha1(x, x, x, mask)  # 
        self_attn = self.dropout1(self_attn, training=training)
        self_attn = self.layernorm1(self_attn + x)
        
        # Q, K, V order for cross_attn inputs: only Q comes from within the decoder
        cross_attn = self.mha2(self_attn, enc_output, enc_output, mask)
        cross_attn = self.dropout2(cross_attn, training=training)
        cross_attn = self.layernorm2(cross_attn + self_attn)
        
        # feed-forward part - first expanding to DFF, then brining back to D_MODEL
        output = self.dense_relu(cross_attn)
        output = self.linear(output)
        
        output = self.dropout3(output, training=training) 
        output = self.layernorm3(output + cross_attn)  # add residual connection and normalize
        
        return output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, window, n_layers, d_model, n_heads, dff, dropout_rate):
        super().__init__()
        self.n_layers = n_layers

        self.dropout = Dropout(dropout_rate)        
        self.enc_layers = [EncoderLayer(d_model, n_heads, dff, dropout_rate) for _ in range(n_layers)]

    def call(self, x, mask, training):

        x = self.dropout(x, training=training)

        for i in range(self.n_layers):  
            x = self.enc_layers[i](x, mask, training)

        return x  # (batch_size, seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, 
                 window, n_layers, d_model, n_heads, dff, dropout_rate):
        super().__init__()

        self.n_layers = n_layers
        self.dropout = Dropout(dropout_rate)        
        self.dec_layers = [DecoderLayer(d_model, n_heads, dff, dropout_rate) for _ in range(n_layers)]
       
    def call(self, x, enc_output, mask, training):

        x = self.dropout(x, training=training)

        for i in range(self.n_layers):  
            x = self.dec_layers[i](x, enc_output, mask, training)

        return x # (batch_size, seq_len, d_model)        


class Transformer(tf.keras.Model):
    """Try symmetric Enc-Dec architecture
    """
    def __init__(self, 
                 window, n_layers, d_model, n_heads, dff, dropout_rate,
                 n_questions, n_lagbins, n_histbins, po):

        super().__init__()

        self.window = window
        self.d_model  = d_model
        self.po = po

        self.emb = Embedding(n_questions+1,   d_model-23, mask_zero=True)  

        self.emb_lag = Embedding(n_lagbins+1, 4)
        self.pos_scaffold = Embedding(2, 16)   # >>>> a temporary solution

        self.encoder = Encoder(window, n_layers, d_model, n_heads, dff, dropout_rate)
        self.decoder = Decoder(window, n_layers, d_model, n_heads, dff, dropout_rate)

        self.final_sigmoid = Dense(1, activation='sigmoid')

        
    def call(self, batch, training):

        batch = tf.cast(batch, tf.float32)
        
        mask = decoder_mask(batch[:,:,0])  

        # ['question_id', 'lag', 'tsl', 'repeat', 'qcp', 'y_roll', 'y_true']
        
        scaff = self.pos_scaffold(batch[:,:, self.po.y_roll])  # batch, seq, 16
        pos = scaff-scaff+positional_encoding()

        question_id = batch[:,:, self.po.question_id]
        question_id = self.emb(question_id)  
        # question_id = question_id + pos

        lag = batch[:,:, self.po.lag]   # embedded lag bins
        lag = self.emb_lag(lag) 

        # tsl = batch[:,:, self.po.tsl]   # embedded lag bins
        # tsl = self.emb_tsl(tsl) 

        repeat = batch[:,:, self.po.repeat]
        repeat = tf.expand_dims(repeat, axis=2)

        qcp = batch[:,:, self.po.qcp]
        qcp = tf.expand_dims(qcp, axis=2)   # (batch_size, seq_len, 1)

        y_roll = batch[:,:, self.po.y_roll]
        y_roll = tf.expand_dims(y_roll, axis=2)   # (batch_size, seq_len, 1)
        
        # Concatenate (NB not add) all inputs:
        # - positional encoding
        # - embedded question id
        # - repeat question indicator
        # - embedded lag
        # - qcp: question correctness percentage
        # - y_roll: previous timestep ground truth 
        x = tf.concat([pos, question_id, repeat, lag, qcp, y_roll], axis=2)

        enc_output = self.encoder(x, mask, training)  # (batch_size, inp_seq_len, d_model)
        
        dec_output = self.decoder(x, enc_output, mask, training) # (batch_size, tar_seq_len, d_model)
        
        output = self.final_sigmoid(dec_output)

        return   tf.squeeze(output)  # (batch_size, tar_seq_len)
