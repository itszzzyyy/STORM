#
#
# """This class handles the creation of the EncoderDecoder model"""
# import tensorflow as tf
# from tensorflow.keras.layers import Attention, Bidirectional, Concatenate, Dense,MultiHeadAttention
# from tensorflow.keras.layers import Embedding, GRU, Input, Lambda, Masking,Layer
# from tensorflow.keras.layers import Permute, TimeDistributed
# from tensorflow.keras.models import Model
# import tensorflow.keras.backend as K
# import numpy as np
#
#
# # def construct_time_matrix(time_data):
# #     """
# #     构建时间间隔矩阵
# #     :param time_data: 形状为 (batch_size, seq_len) 的时间数据，单位为分钟
# #     :return: 时间间隔矩阵，形状为 (batch_size, seq_len, seq_len)
# #     """
# #     # time_data 是 (batch_size, seq_len)
# #     # 扩展维度使其广播为 (batch_size, seq_len, 1) 和 (batch_size, 1, seq_len)
# #     time_i = tf.expand_dims(time_data, axis=2)  # (batch_size, seq_len, 1)
# #     time_j = tf.expand_dims(time_data, axis=1)  # (batch_size, 1, seq_len)
# #
# #     # 计算时间差矩阵，得到 (batch_size, seq_len, seq_len)
# #     delta_matrix = tf.abs(time_i - time_j)
# #
# #     return delta_matrix
# #
# class MHA(Model):
#     def __init__(self, num_heads,  src_traj_len,trg_traj_len,dim_out,d_model=512, attn_drop=0.0, proj_drop=0.0,
#                   add_temporal_bias=True,
#                  temporal_bias_dim=64, use_mins_interval=True):
#         super(MHA, self).__init__()
#         assert d_model % num_heads == 0
#
#         # We assume d_v always equals d_k
#         self.d_model = d_model
#         self.d_k = d_model // num_heads
#         self.num_heads = num_heads
#         # self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
#         self.add_temporal_bias = add_temporal_bias
#         self.temporal_bias_dim = temporal_bias_dim
#         self.use_mins_interval = use_mins_interval
#
#         # 创建三个线性层 (相当于 PyTorch 中的 nn.Linear)
#         self.query_layer = tf.keras.layers.Dense(d_model,use_bias=False)
#         self.key_layer = tf.keras.layers.Dense(d_model,use_bias=False)
#         self.value_layer = tf.keras.layers.Dense(d_model,use_bias=False)
#
#         # Dropout 层
#         self.dropout = tf.keras.layers.Dropout(rate=attn_drop)
#
#         # 最终投影层
#         self.proj = tf.keras.layers.Dense(d_model,use_bias=False)
#         self.proj_drop = tf.keras.layers.Dropout(rate=proj_drop)
#
#     # def build(self, input_shape):
#     #     # input_shape 是一个包含两个输入形状的元组，例如 (query_shape, value_shape)
#     #     query_shape, value_shape = input_shape
#     #
#     #     # 保存输入形状以备后用
#     #     self.query_shape = query_shape
#     #     self.value_shape = value_shape
#     #     # 调用父类的 build 方法
#     #     super(MHA, self).build(input_shape)
#
#     def get_config(self):
#         config = super(MHA, self).get_config()  # 获取父类的配置
#         config.update({
#             'd_model': self.d_model,
#             'num_heads': self.num_heads,
#             'd_model': self.d_k * self.num_heads,
#             'src_traj_len': self.src_traj_len,
#             'trg_traj_len': self.trg_traj_len,
#             'dim_out': self.d_model,
#             'attn_drop': self.dropout.rate,
#             'proj_drop': self.proj_drop.rate,
#             'add_temporal_bias': self.add_temporal_bias,
#             'temporal_bias_dim': self.temporal_bias_dim,
#             'use_mins_interval': self.use_mins_interval,
#         })
#         return config
#     def split_heads(self, x, batch_size):
#         """
#         将最后一个维度分割成 (num_heads, depth).
#         转置为 (batch_size, num_heads, seq_len, depth)
#         """
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
#         return tf.transpose(x, perm=[0, 2, 1, 3])
#
#     def call(self, query_input, value_input, output_attentions=False, batch_temporal_mat=None):
#         batch_size = tf.shape(query_input)[0]
#         # 1) 线性变换
#         query = self.query_layer(query_input)  # (B, T_q, d_model)
#         key = self.key_layer(value_input)  # (B, T_v, d_model)
#         value = self.value_layer(value_input)  # (B, T_v, d_model)
#         # 2) 分头
#         query = self.split_heads(query, batch_size)  # (B, num_heads, T_q, d_k)
#         key = self.split_heads(key, batch_size)  # (B, num_heads, T_v, d_k)
#         value = self.split_heads(value, batch_size)  # (B, num_heads, T_v, d_k)
#
#         # 3) 计算注意力得分
#         scores = tf.matmul(query, key, transpose_b=True) # (B, num_heads, T_q, T_v)
#         print("注意力得分形状",scores.shape)
#
#         # # 4) 添加时间偏置（如果需要）
#         # if self.add_temporal_bias and batch_temporal_mat is not None:
#         #     if self.use_mins_interval:
#         #         batch_temporal_mat = 1.0 / tf.math.log(
#         #             tf.math.exp(1.0) + (batch_temporal_mat / 60.0))
#         #     else:
#         #         batch_temporal_mat = 1.0 / tf.math.log(tf.math.exp(1.0) + batch_temporal_mat)
#         #
#         #     if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
#         #         batch_temporal_mat = self.temporal_mat_bias_2(
#         #             tf.nn.leaky_relu(
#         #                 self.temporal_mat_bias_1(tf.expand_dims(batch_temporal_mat, -1)), alpha=0.2
#         #             )
#         #         )
#         #         batch_temporal_mat = tf.squeeze(batch_temporal_mat, -1)
#         #
#         #     if self.temporal_bias_dim == -1:
#         #         batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias
#         #
#         #     batch_temporal_mat = tf.expand_dims(batch_temporal_mat, 1)  # (B, 1, T_q, T_v)
#         #     scores += batch_temporal_mat
#
#         # 5) 计算注意力权重
#         attn_weights = tf.nn.softmax(scores, axis=-1)  # (B, num_heads, T_q, T_v)
#         attn_weights = self.dropout(attn_weights)
#
#         # 6) 计算输出
#         output = tf.matmul(attn_weights, value)  # (B, num_heads, T_q, d_k)
#         output = tf.transpose(output, perm=[0, 2, 1, 3])  # (B, T_q, num_heads, d_k)
#         output = tf.reshape(output, (batch_size, -1, self.d_model))  # (B, T_q, d_model)
#
#         # 7) 最终投影
#         output = self.proj(output)  # (B, T_q, dim_out)
#         output = self.proj_drop(output)
#
#         if output_attentions:
#             return output, attn_weights  # 返回输出和注意力权重
#         else:
#             return output
#
#
# class LearnableFourierPositionalEncoding:
#     def __init__(self, M, F_dim, D, gamma=1.0,names = 'LearnableFourierPositionalEncoding'):
#
#         """
#         Initialize the LearnableFourierPositionalEncoding model.
#
#         :param M: Each point has an M-dimensional positional value
#         :param F_dim: Depth of the Fourier feature dimension
#         :param D: Positional encoding dimension
#         :param gamma: Parameter to initialize Wr
#         """
#         # Store parameters
#         self.M = M
#         self.F_dim = F_dim
#         self.D = D
#         self.gamma = gamma
#
#         # Define the input layer
#         inputs = Input(shape=(None, M))
#
#         # Define the projection layer (equivalent to a Dense layer without bias)
#         Wr = Dense(self.F_dim // 2, use_bias=False)
#
#         # Define the MLP as a Sequential model
#         mlp = tf.keras.Sequential([
#             Dense(D, activation='gelu'),
#             Dense(D)
#         ])
#
#         # Define the custom Fourier positional encoding layer using Lambda
#         def positional_encoding_layer(x):
#             projected = Wr(x)  # Linear transformation
#             cosines = tf.math.cos(projected)
#             sines = tf.math.sin(projected)
#             F = 1 / tf.math.sqrt(tf.cast(self.F_dim, tf.float32)) * tf.concat([cosines, sines], axis=-1)
#             Y = mlp(F)
#             # pos_enc = Y.reshape((B, N, self.D))
#             return Y
#
#         # Apply the custom layer using Lambda
#         outputs = positional_encoding_layer(inputs)
#
#         # Build the model
#         self.model = Model(inputs=inputs, outputs=outputs, name=names)
#         # self.model.summary()
#
#
# # def summary(self):
#     #     """Print the summary of the model."""
#     #     self.model.summary()
#
# #
#
# class StackedGRU():
#     """
#     Handles the Encoder part of the overall encoder-decoder model. The Encoder
#     takes as input the trajectory after the embedding is done and outputs the
#     hidden state of each layer as well as the feature vector representation of
#     each trajectory after being passed through the GRU layers contained within.
#     """
#
#     def __init__(self, embedding_size, gru_cell_size, num_gru_layers,
#                  gru_dropout_ratio, bidirectional, encoder_h0):
#         """
#         Initializes the model.
#
#         Inputs:
#             input: Shape is (batch_size, in_traj_len, embedding_size) This is
#                    the trajectory after each of its spatiotemporal cells have
#                    been embedded. in_traj_len depends on the input tensor
#         Outputs
#             hn: Shape is (batch_size,num_gru_layers * num_directions,
#                 gru_cell_size). This is the hidden state of each of the GRU
#                 layers
#             output: Shape is (batch_size, in_traj_len, gru_cell_size *
#                     num_directions). This is the encoded input trajectory.
#         Where:
#             - batch_size: Size of input batch
#             - in_traj_len: Length of the input trajectory
#             - embedding_size: The size of the embedding
#             - num_gru_layers: The number of GRU layers
#             - num_directions: This is 1 if bidirectional is False. 2 if True.
#             - gru_cell_size: The size of the GRU cells
#
#         Arguments:
#             embedding_size: (integer) The size of the spatiotemporal cells embedding
#             gru_cell_size: (integer) Size of the gru cels
#             num_gru_layers: (integer) How many GRus to use
#             gru_dropout_ratio: (float) The dropout ratio for the GRUs
#             bidirectional: (boolean) Whether or not to use bidirectional GRUs
#             encoder_h0: (tensor) Initial state of the GRU
#         """
#         # (batch_size, in_traj_len, embedding_size)
#         # in_traj_len is not static
#         inputs = Input((None, embedding_size))
#
#         hn = []
#         if bidirectional:
#             gru = Bidirectional(GRU(gru_cell_size,
#                                     dropout = gru_dropout_ratio,
#                                     return_sequences = True))\
#                                (inputs, initial_state = encoder_h0)
#             hn.append(gru)
#             for i in range(num_gru_layers-1):
#                 gru = Bidirectional(GRU(gru_cell_size,
#                                         return_sequences = True,
#                                         dropout = gru_dropout_ratio))(gru)
#                 hn.append(gru)
#         else:
#             gru = GRU(gru_cell_size, return_sequences = True,
#                       dropout = gru_dropout_ratio)\
#                       (inputs, initial_state=encoder_h0)
#             hn.append(gru)
#             for i in range(num_gru_layers-1):
#                 gru = GRU(gru_cell_size, return_sequences = True,
#                           dropout = gru_dropout_ratio)(gru)
#                 hn.append(gru)
#         model = Model(inputs = inputs, outputs = gru,name='GRU')
#         self.model = model
#
#
# class Encoder():
#     """
#     Handles the Encoder part of the overall encoder-decoder model. The Encoder
#     takes as input the trajectory after the embedding is done and outputs the
#     hidden state of each layer as well as the feature vector representation of
#     each trajectory after being passed through the GRU layers contained within.
#     """
#
#     def __init__(self, in_traj_len, embedding, gru):
#         """
#         Initializes the model.
#
#         Args:
#             in_traj_len: (integer) Length of trajectory
#             embedding: (keras model) The embedding layer
#             gru: (keras model) The stacked GRU model
#         """
#         inputs = Input(shape=(None, ))  # 假设 M 是输入的最后一个维度
#         # print(inputs.shape)
#         # positional_encoding = LearnableFourierPositionalEncoding(M=1, F_dim=256, D=256)
#         # encoded_positions = positional_encoding.model(inputs)
#         # print(encoded_positions.shape)
#         embedded = embedding(inputs)
#         # print(embedded.shape)
#         output = gru(embedded)
#         outputs = Attention()([output])
#         model = Model(inputs = inputs, outputs = outputs)
#         self.model = model
#         # self.model.summary()
#
#
# class query_Encoder():
#     """
#     Handles the Encoder part of the overall encoder-decoder model. The Encoder
#     takes as input the trajectory after the embedding is done and outputs the
#     hidden state of each layer as well as the feature vector representation of
#     each trajectory after being passed through the GRU layers contained within.
#     """
#
#     def __init__(self, in_traj_len, embedding, gru):
#         """
#         Initializes the model.
#
#         Args:
#             in_traj_len: (integer) Length of trajectory
#             embedding: (keras model) The embedding layer
#             gru: (keras model) The stacked GRU model
#         """
#         inputs_1 = Input((None, ),name='query')
#         inputs_2 = Input((None, 2 ),name='spa')
#         # inputs_3 = Input((None, ),name='longi')
#         inputs_3 = Input((None, ),name='tem')
#         print("inputs_1: ",inputs_1.shape)
#         print("inputs_2: ",inputs_2.shape)
#         # print("inputs_3: ",inputs_3.shape)
#         # print("inputs_4: ",inputs_4.shape)
#         # lat_encoding = LearnableFourierPositionalEncoding(M=1, F_dim=256, D=256,names = 'learnable1')
#         spa_encoding = LearnableFourierPositionalEncoding(M=2, F_dim=256, D=256,names = 'learnable2')
#         temporal_encoding = LearnableFourierPositionalEncoding(M=1, F_dim=256, D=256, names = 'learnable3')
#         encoded_spa = spa_encoding.model(inputs_2)
#         # encoded_lon = lon_encoding.model(inputs_3)
#         encoded_temporal = temporal_encoding.model(inputs_3)
#         embeddings = embedding(inputs_1)
#         # print("encoded_positions: ",encoded_positions.shape)
#         print("encoded_temporal: ",encoded_temporal.shape)
#         print("embeddings: ",embeddings.shape)
#         enc = encoded_spa + encoded_temporal + embeddings
#         print("enc: ",enc.shape)
#
#         # embedded = embedding(encoded_positions)
#         # print(embedded.shape)
#         outputs = gru(enc)
#         model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
#         self.model = model
#         # self.model.summary()
# class Decoder():
#     """
#     Handles the Decoder part of the overall encoder-decoder model. The Encoder
#     first produces the feature vector representation of the target
#     trajectory. Then, an attention layer is applied to this feature vector
#     representation and the source trajectory feature vector representation.
#     The output of this attention module is used as the Decoder
#     output.
#     """
#
#     def __init__(self, src_traj_len, src_feature_size, trg_traj_len,
#                   embedding, gru,use_attention=True):
#         """
#         Initialize the model
#
#         Inputs:
#             inputs_1: Shape is (batch_size, src_traj_len, src_feature_size).
#                       This is the learned representation of the source
#                       trajectory, which is the output from the Encoder
#             inputs_2: Shape is (batch_size, trg_traj_len). This is the target
#                       trajectory.
#
#         Outputs:
#             outputs: Shape is (batch_size, trg_traj_len, src_feature_size)
#
#         Args:
#             src_traj_len: (integer) The length of the source trajectory
#             src_feature_size: (integer) Feature vector size of each sequence in
#                                the target trajectory.
#             trg_traj_len: (integer) The length of the target trajectory
#             use_attention: (boolean) Whether or not to use the attention module
#             embedding: (keras model) The embedding layer
#             gru: (keras model) The GRU model
#         """
#         # embedded_src (batch_size, src_traj_len, src_feature_size)
#         # This input is for the encoded source trajectory
#         # self.attention = MHA(num_heads=8, d_model=src_feature_size, src_traj_len=src_traj_len,trg_traj_len=trg_traj_len,dim_out=src_feature_size)
#
#         embedded_src = Input((src_traj_len, src_feature_size))
#
#         # inputs_trg (batch_size, trg_traj_len)
#         # This input is for the target trajectory
#         inputs_trg = Input((trg_traj_len,))
#         # time_matrix = Input((src_traj_len,))
#
#         # inputs_trg (batch_size, trg_traj_len)
#         # embedded_trg (batch_size, trg_traj_len, src_feature_size)
#         # Since the embedding layer used is the same as the target, the
#         # output feature size is the same as src_feature_size
#         embedded_trg = embedding(inputs_trg)
#         embedded_trg = gru(embedded_trg)
#
#         # embedded_src (batch_size, src_traj_len, src_feature_size)
#         # embedded_trg (batch_size, trg_traj_len, src_feature_size)
#         # outputs (batch_size, trg_traj_len, src_feature_size)
#         # TODO: OPTION TO NOT USE ATTENTION
#         print("embedded_trg", embedded_trg.shape)
#         print("embedded_src", embedded_src.shape)
#
#         outputs = Attention()([embedded_trg, embedded_src])
#         self.model = Model(inputs = [embedded_src, inputs_trg],outputs=outputs)
#         self.model.summary()
#
#
# class PatternDecoder():
#     """
#     The pattern decoder. It first takes the target pattern and produces a
#     feature vector representation so that the dimension matches the source
#     input, which is the source trajectory feature vector representation. Then,
#     the two are fed into an attention layer to produce the prediction
#     """
#     def __init__(self, src_traj_len, src_feature_size, trg_traj_len,use_attention=True
#                 ):
#         """
#         Initialize the model
#
#         Inputs:
#             inputs_1: Shape is (batch_size, src_traj_len, src_feature_size).
#                       This is the learned representation of the source
#                       trajectory, which is the output from the Encoder
#             inputs_2: Shape is (batch_size, trg_traj_len, 2). This is the
#                       target trajectory pattern features, which includes the
#                       spatial and temporal features.
#
#         Outputs:
#             outputs: Shape is (batch_size, trg_traj_len, src_feature_size)
#
#         Args:
#             src_traj_len: (integer) The length of the source trajectory
#             src_feature_size: (integer) Feature vector size of each sequence in
#                                the target trajectory.
#             trg_traj_len: (integer) The length of the target trajectory
#             use_attention: (boolean) Whether or not to use the attention module
#         """
#         # embedded_src (batch_size, src_traj_len, src_feature_size)
#         # This input is for the encoded source trajectory
#         # self.attention = MHA(num_heads=8, d_model=src_feature_size, dim_out=src_feature_size)
#         embedded_src = Input((src_traj_len, src_feature_size))
#
#         # inputs_trg (batch_size, trg_traj_len, 2)
#         # This input is for the target spatiotemporal trajectory pattern
#         inputs_trg = Input((trg_traj_len, 2))
#
#         # inputs_trg (batch_size, trg_traj_len, 2)
#         # embedded_trg (batch_size, trg_traj_len, src_feature_size)
#         # Learns features based on inputs_trg.
#         embedded_trg = TimeDistributed(Dense(src_feature_size,
#                                              activation = 'relu'))(inputs_trg)
#
#         # embedded_src (batch_size, src_traj_len, src_feature_size)
#         # embedded_trg (batch_size, trg_traj_len, src_feature_size)
#         # outputs (batch_size, trg_traj_len, src_feature_size)
#         # TODO: OPTION TO NOT USE ATTENTION
#         outputs = Attention()([embedded_trg, embedded_src])
#         # outputs = self.attention(embedded_src, embedded_trg)
#         print("outputs", outputs.shape)
#         # outputs = Multiattention(num_heads=8, output_dim=src_feature_size, key_dim=src_feature_size )(
#         #     [embedded_src, embedded_trg])
#         self.model = Model(inputs = [embedded_src, inputs_trg],outputs=outputs)
#         self.model.summary()
#
#
# class EncoderDecoder():
#     """
#     The EncoderDecoder model is the final model to be used, which is an
#     amalgamation of the Encoder, Decoder, and other components contained
#     within the other classes in this module.
#     """
#
#     def __init__(self, src_traj_len, trg_traj_len, trg_patt_len,
#                  embed_vocab_size, embedding_size, gru_cell_size,
#                  num_gru_layers, gru_dropout_ratio, bidirectional,
#                  use_attention,
#                  encoder_h0 = None):
#         """
#         Initializes the model.
#
#         Inputs:
#             input: Shape is (batch_size, src_traj_len). This is the input
#                    trajectory where each item in the sequence is the
#                    spatiotemporal cell IDs
#         Outputs:
#             output: Shape is (batch_size, trg_traj_len, embedding_size). It
#                     contains the feature vector representation of the predicted
#                     output trajectory.
#         Where:
#             - batch_size: Size of input batch
#             - src_traj_len: Length of the source (i.e. query) trajectory
#             - trg_traj_len: Length of the output (i.e. ground truth) trajectory
#             - k: Stands for the top-k nearest neighbors' weights.
#
#         Args:
#             src_traj_len: (integer) Length of source trajectory
#             trg_traj_len: (integer) Length of target trajectory
#             embed_vocab_size: (integer) Size of the embedding layer vocabulary
#             embedding_size: (integer) Size of the embedding layer output
#             gru_cell_size: (integer) Size of the gru cels
#             num_gru_layers: (integer) How many GRUs to use
#             gru_dropout_ratio: (float) Dropout ratio for the GRUs
#             bidirectional: (boolean) Whether or not bidirectional GRU is used
#             use_attention: (boolean) Whether or not the attention model is used
#             encoder_h0: (tensor) Initial state of the encoder
#         """
#         # inputs_1 (batch_size, src_traj_len)
#         # This input is for the source trajectory
#         inputs_1 = Input((src_traj_len,),name='q')
#         print("src_traj shape", inputs_1.shape)
#
#         # inputs_2 (batch_size, trg_traj_len)
#         # This input is for the target trajectory
#         inputs_2 = Input((trg_traj_len,),name='gt')
#
#         # inputs_3 (batch_size, trg_patt_len)
#         # This input is for the target pattern
#         inputs_3 = Input((trg_patt_len,2),name='patt')
#         inputs_4 = Input((src_traj_len,2),name='spa')
#         inputs_5 = Input((src_traj_len, ),name='tem')
#         # This is for MHA
#         # inputs_6 = Input((src_traj_len, ),name='temmatrix')
#
#         # Encoder part
#         self.sgru = StackedGRU(embedding_size, gru_cell_size,
#                                num_gru_layers, gru_dropout_ratio, bidirectional,
#                                encoder_h0).model
#         self.embedding = Embedding(embed_vocab_size, embedding_size)
#         self.encoder = Encoder(src_traj_len,self.embedding,self.sgru)
#         self.q_encoder = query_Encoder(src_traj_len, self.embedding, self.sgru)
#         # inputs (batch_size, src_traj_len)
#         # encoded (batch_size, src_traj_len, gru_cell_size * directions)
#         # directions = 2 if bidirectional, else 1.
#         encoded = self.q_encoder.model([inputs_1,inputs_4,inputs_5])
#
#         # Decoder part
#         traj_repr_size = gru_cell_size
#         if bidirectional:
#             traj_repr_size *= 2
#         self.decoder = Decoder(src_traj_len, traj_repr_size, trg_traj_len,
#                                self.embedding, self.sgru,use_attention)
#         # encoded (batch_size, src_traj_len, gru_cell_size * directions)
#         # inputs_2 (batch_size, trg_traj_len)
#         # output_point (batch_size, trg_traj_len, gru_cell_size * directions)
#         output_point = self.decoder.model([encoded, inputs_2])
#
#         # Pattern decoder part
#         # Pattern decoder
#         self.patt_decoder = PatternDecoder(src_traj_len, traj_repr_size,
#                                            trg_patt_len,use_attention)
#         # encoded (batch_size, src_traj_len, embedding_size)
#         # inputs_2 (batch_size, trg_traj_len)
#         # output_patt (batch_size, trg_patt_len, embedding_size)
#         output_patt = self.patt_decoder.model([encoded, inputs_3])
#         print("out_point", output_point.shape)
#         print("output_patt", output_patt.shape)
#
#         # Finished model
#         self.model = Model(inputs = [inputs_1, inputs_2, inputs_3, inputs_4, inputs_5],
#                            outputs = [output_point, output_patt])
#         self.model.summary()
#
#
# class STSeqModel():
#     """This class handles the SpatioTemporal Sequence2Sequence """
#
#
#     def __init__(self, embed_vocab_size, embedding_size, traj_repr_size,
#                  gru_cell_size, num_gru_layers, gru_dropout_ratio,
#                  bidirectional, k,use_attention):
#         """
#         Creates the model
#
#         Args:
#             gru_cell_size: (integer) Size of every LSTM in the model
#             traj_repr_size: (integer) The size of the trajectory vector
#                              representation
#             xshape: (numpy array) The size of the input numpy arrays
#         """
#         # 'inputs' shape (batch_size, num_inner_data, traj_len, 1)
#         # num_inner_data should be 7, representing:
#         # - ground truth trajectory,
#         # - query trajectory
#         # - negative trajectory
#         # - spatial pattern
#         # - temporal pattern
#         # - jing du
#         # - wei du
#         # - shi jian
#         self.__NUM_FEATURES = 14
#         self.__NUM_INNER_FEATURES = 1
#         inputs = Input((self.__NUM_FEATURES, None, self.__NUM_INNER_FEATURES))
#
#         ## Lambda layers to split the inputs.
#         # 'gt' shape (batch_size, traj_len, 1).
#         # Represents ground truth trajectory.
#         gt = Lambda(lambda x:x[:,0,:,0])(inputs)
#         gt = Masking(mask_value = 0)(gt)
#
#         # 'q' shape (batch_size, traj_len, 1).
#         # Represents the query trajectory.
#         q = Lambda(lambda x:x[:,1,:,0])(inputs)
#         q = Masking(mask_value = 0)(q)
#         print("q.shape", q.shape)
#
#         # 'neg' shape (batch_size, traj_len, 1).
#         # Represents the negative trajectory.
#         neg = Lambda(lambda x:x[:,2,:,0])(inputs)
#         neg = Masking(mask_value = 0)(neg)
#
#         # 'gt_patt_s' shape (batch_size, traj_len, 1).
#         gt_patt_s = Lambda(lambda x:x[:,3,:,:])(inputs)
#         gt_patt_s = Masking(mask_value = 0)(gt_patt_s)
#
#         # 'gt_patt_t' shape (batch_size, traj_len, 1).
#         gt_patt_t = Lambda(lambda x:x[:,4,:,:])(inputs)
#         gt_patt_t = Masking(mask_value = 0)(gt_patt_t)
#         q_lat =  Lambda(lambda x:x[:,5,:,0])(inputs)
#         q_lat = Masking(mask_value = 0)(q_lat)
#         print('q_lat.shape',q_lat.shape)
#         q_lon = Lambda(lambda x:x[:,6,:,0])(inputs)
#         q_lon = Masking(mask_value=0)(q_lon)
#         print('q_lon.shape',q_lon.shape)
#         q_t = Lambda(lambda x:x[:,7,:,0])(inputs)
#         q_t = Masking(mask_value=0)(q_t)
#         print('q_t.shape',q_t.shape)
#         # batch_time_temporal = construct_time_matrix(q_t)
#         # print("batch_time_temporal",batch_time_temporal.shape)
#         gt_lat = Lambda(lambda x:x[:,8,:,0])(inputs)
#         gt_lat = Masking(mask_value=0)(gt_lat)
#         gt_lon = Lambda(lambda x: x[:, 9, :, 0])(inputs)
#         gt_lon = Masking(mask_value=0)(gt_lon)
#         gt_t = Lambda(lambda x: x[:, 10, :, 0])(inputs)
#         gt_t = Masking(mask_value=0)(gt_t)
#         neg_lat = Lambda(lambda x: x[:, 11, :, 0])(inputs)
#         neg_lat = Masking(mask_value=0)(neg_lat)
#         neg_lon = Lambda(lambda x: x[:, 12, :, 0])(inputs)
#         neg_lon = Masking(mask_value=0)(neg_lon)
#         neg_t = Lambda(lambda x: x[:, 13, :, 0])(inputs)
#         neg_t = Masking(mask_value=0)(neg_t)
#         q_lat_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(q_lat)  # 形状变为 (batch_size, None, 1)
#         q_lon_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(q_lon)  # 形状变为 (batch_size, None, 1)
#         #
#         # # 在特征维度 (axis=2) 上合并它们
#         q_spa = Concatenate(axis=-1)([q_lat_expanded, q_lon_expanded])  # 合并后的形状是 (batch_size, None, 2)
#         gt_lat_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(gt_lat)  # 形状变为 (batch_size, None, 1)
#         gt_lon_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(gt_lon)  # 形状变为 (batch_size, None, 1)
#         #
#         # # 在特征维度 (axis=2) 上合并它们
#         gt_spa = Concatenate(axis=-1)([gt_lat_expanded, gt_lon_expanded])  # 合并后的形状是 (batch_size, None, 2)
#         neg_lat_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(neg_lat)  # 形状变为 (batch_size, None, 1)
#         neg_lon_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(neg_lon)  # 形状变为 (batch_size, None, 1)
#         #
#         # # 在特征维度 (axis=2) 上合并它们
#         neg_spa = Concatenate(axis=-1)([neg_lat_expanded, neg_lon_expanded])  # 合并后的形状是 (batch_size, None, 2)
#         # 'gt_patt_st' shape (batch_size, traj_len, 2)
#         gt_patt_st = Concatenate(axis=2)([gt_patt_s, gt_patt_t])
#         print("q shape: ", q.shape)
#         # print("q_spa shape: ", q_spa.shape)
#         print("q_t shape: ", q_t.shape)
#         print("gt_spa shape: ", gt_spa)
#         print("neg_spa shape: ", neg_spa)
#
#         # EncoderDecoder model
#         assert gt_patt_s.shape[0] == gt_patt_t.shape[0]
#         src_traj_len = q.shape[1]
#         trg_traj_len = gt.shape[1]
#         trg_patt_len = gt_patt_s.shape[1]
#         # st_len = q_t.shape[1]
#
#         # Getting the point-to-point and pattern representation
#         # Inputs:
#         # 'q' shape (batch_size, traj_len, 1).
#         # 'gt' shape (batch_size, traj_len, 1).
#         # 'gt_patt_st' shape (batch_size, traj_len, 2)
#         # Outputs:
#         # 'traj_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
#         # 'patt_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
#         # directions = 2 if bidirectional, else 1
#         encoder_decoder = EncoderDecoder(src_traj_len, trg_traj_len,
#                                          trg_patt_len, embed_vocab_size,
#                                          embedding_size, gru_cell_size,
#                                          num_gru_layers, gru_dropout_ratio,
#                                          bidirectional,use_attention)
#         [traj_repr, patt_repr] = encoder_decoder.model([q, gt, gt_patt_st,q_spa,q_t])
#
#         # Encoder part
#         self.encoder = encoder_decoder.encoder
#         self.q_encoder = encoder_decoder.q_encoder
#
#         # Getting the trajectory representation
#         # Inputs:
#         # 'q' shape (batch_size, traj_len, 1).
#         # 'gt' shape (batch_size, traj_len, 1).
#         # 'neg' shape (batch_size, traj_len, 1).
#         # Outputs:
#         # 'enc_q' shape (batch_size, traj_len, gru_cell_size * directions).
#         # 'enc_gt' shape (batch_size, traj_len, gru_cell_size * directions).
#         # 'enc_neg' shape (batch_size, traj_len, gru_cell_size * directions).
#         # directions = 2 if bidirectional, else 1
#         # q = Input(shape=(src_traj_len,),dtype='float32')
#         # q_spa = Input(shape=(src_traj_len,2),dtype='float32')
#         # # q_spa = Masking(mask_value=0)(q_spa)
#         # q_t = Input(shape=(src_traj_len,1),dtype='float32')
#         # print("q shape: ", q.shape)
#         # print("q_spa shape: ", q_spa.shape)
#         # print("q_t shape: ", q_t.shape)
#         # print("gt shape: ", gt)
#         # print("neg shape: ", neg)
#
#         enc_gt = self.q_encoder.model([gt,gt_spa,gt_t])
#         enc_neg = self.q_encoder.model([neg,neg_spa,neg_t])
#         enc_q = self.q_encoder.model([q, q_spa, q_t])
#
#         # Three loss functions needed, so we need three outputs
#         # First is the representation loss which takes the encoder outputs
#         # Inputs:
#         # 'enc_q' shape (batch_size, traj_len, gru_cell_size * directions).
#         # 'enc_gt' shape (batch_size, traj_len, gru_cell_size * directions).
#         # 'enc_neg' shape (batch_size, traj_len, gru_cell_size * directions).
#         # Outputs:
#         # 'out_repr' shape (batch_size, 3, traj_len, gru_cell_size * directions)
#         out_repr = K.stack([enc_q, enc_gt, enc_neg], axis=1)
#
#         # Second is the point-to-point loss.
#         # Inputs:
#         # 'traj_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
#         # Outputs:
#         # 'out_traj' shape (batch_size, trg_traj_len, k)
#         out_traj = TimeDistributed(Dense(k, activation = 'relu'))(traj_repr)
#
#         # Third is the pattern loss
#         # Inputs:
#         # 'patt_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
#         # Outputs:
#         # 'out_patt' shape (batch_size, trg_traj_len, 2)
#         out_patt = TimeDistributed(Dense(2, activation = 'relu'))(patt_repr)
#
#         # Create model
#         model = Model(inputs = inputs, outputs = [out_repr, out_traj, out_patt])
#         self.model = model
#         self.model.summary()


"""This class handles the creation of the EncoderDecoder model"""
import tensorflow as tf
from tensorflow.keras.layers import Attention, Bidirectional, Concatenate, Dense,MultiHeadAttention
from tensorflow.keras.layers import Embedding, GRU, Input, Lambda, Masking,Layer
from tensorflow.keras.layers import Permute, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np


def construct_time_matrix(time_data):
    """
    构建时间间隔矩阵
    :param time_data: 形状为 (batch_size, seq_len) 的时间数据，单位为分钟
    :return: 时间间隔矩阵，形状为 (batch_size, seq_len, seq_len)
    """
    # time_data 是 (batch_size, seq_len)
    # 扩展维度使其广播为 (batch_size, seq_len, 1) 和 (batch_size, 1, seq_len)
    time_i = tf.expand_dims(time_data, axis=2)  # (batch_size, seq_len, 1)
    time_j = tf.expand_dims(time_data, axis=1)  # (batch_size, 1, seq_len)

    # 计算时间差矩阵，得到 (batch_size, seq_len, seq_len)
    delta_matrix = tf.abs(time_i - time_j)

    return delta_matrix

# class MHA(Model):
#     def __init__(self, num_heads=8,d_model=512, attn_drop=0.0, proj_drop=0.0,
#                   add_temporal_bias=True,
#                  temporal_bias_dim=64, use_mins_interval=False):
#         super(MHA, self).__init__()
#         assert d_model % num_heads == 0
#
#         # We assume d_v always equals d_k
#         self.d_model = d_model
#         self.d_k = d_model // num_heads
#         self.num_heads = num_heads
#         self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
#         self.add_temporal_bias = add_temporal_bias
#         self.temporal_bias_dim = temporal_bias_dim
#         self.use_mins_interval = use_mins_interval
#         # 创建三个线性层 (相当于 PyTorch 中的 nn.Linear)
#         self.query_layer = tf.keras.layers.Dense(units=d_model, input_shape=(None,d_model),use_bias=False)
#         self.key_layer = tf.keras.layers.Dense(units=d_model, input_shape=(None,d_model), use_bias=False)
#         self.value_layer = tf.keras.layers.Dense(units=d_model, input_shape=(None,d_model), use_bias=False)
#         # Dropout 层
#         self.dropout = tf.keras.layers.Dropout(rate=attn_drop)
#
#         # 最终投影层
#         self.proj = tf.keras.layers.Dense(units=d_model,input_shape=(None,d_model),use_bias=False)
#         self.proj_drop = tf.keras.layers.Dropout(rate=proj_drop)
#         if self.add_temporal_bias:
#             if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
#                 self.temporal_mat_bias_1 = tf.keras.layers.Dense(self.temporal_bias_dim, use_bias=False)
#                 self.temporal_mat_bias_2 = tf.keras.layers.Dense(1, use_bias=False)
#             # elif self.temporal_bias_dim == -1:
#             #     self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
#             #     nn.init.xavier_uniform_(self.temporal_mat_bias)
#
#
#
#     # def build(self, input_shape):
#     #     # input_shape 是一个包含两个输入形状的元组，例如 (query_shape, value_shape)
#     #     query_shape, value_shape = input_shape
#     #
#     #     # 保存输入形状以备后用
#     #     self.query_shape = query_shape
#     #     self.value_shape = value_shape
#     #     # 调用父类的 build 方法
#     #     super(MHA, self).build(input_shape)
#
#     def get_config(self):
#         config = super(MHA, self).get_config()  # 获取父类的配置
#         config.update({
#             'd_model': self.d_model,
#             'num_heads': self.num_heads,
#             'd_model': self.d_k * self.num_heads,
#             'src_traj_len': self.src_traj_len,
#             'trg_traj_len': self.trg_traj_len,
#             'dim_out': self.d_model,
#             'attn_drop': self.dropout.rate,
#             'proj_drop': self.proj_drop.rate,
#             'add_temporal_bias': self.add_temporal_bias,
#             'temporal_bias_dim': self.temporal_bias_dim,
#             'use_mins_interval': self.use_mins_interval,
#         })
#         return config
#     def split_heads(self, x, batch_size):
#         """
#         将最后一个维度分割成 (num_heads, depth).
#         转置为 (batch_size, num_heads, seq_len, depth)
#         """
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
#         return tf.transpose(x, perm=[0, 2, 1, 3])
#
#     def call(self, query_input,key_input, output_attentions=False, batch_temporal_mat=None):
#         batch_size = tf.shape(query_input)[0]
#         # 1) 线性变换
#         query = self.query_layer(query_input)  # (B, T_q, d_model)
#         key = self.key_layer(key_input)  # (B, T_v, d_model)
#         value = self.value_layer(key_input)  # (B, T_v, d_model)
#         # 2) 分头
#         query = self.split_heads(query, batch_size)  # (B, num_heads, T_q, d_k)
#         key = self.split_heads(key, batch_size)  # (B, num_heads, T_v, d_k)
#         value = self.split_heads(value, batch_size)  # (B, num_heads, T_v, d_k)
#
#         # 3) 计算注意力得分
#         scores = tf.matmul(query, key, transpose_b=True)*self.scale # (B, num_heads, T_q, T_v)
#         # scores = tf.matmul(query, key, transpose_b=True) * self.scale  # (B, num_heads, T_q, T_v)
#         print("注意力得分形状",scores.shape)
#
#         # # 4) 添加时间偏置（如果需要）
#         if self.add_temporal_bias and batch_temporal_mat is not None:
#             if self.use_mins_interval:
#                 batch_temporal_mat = 1.0 / tf.math.log(
#                     tf.math.exp(1.0) + (batch_temporal_mat / 60.0))
#             else:
#                 batch_temporal_mat = 1.0 / tf.math.log(tf.math.exp(1.0) + batch_temporal_mat)
#
#             if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
#                 batch_temporal_mat = self.temporal_mat_bias_2(
#                     tf.nn.leaky_relu(
#                         self.temporal_mat_bias_1(tf.expand_dims(batch_temporal_mat, -1)), alpha=0.2
#                     )
#                 )
#                 batch_temporal_mat = tf.squeeze(batch_temporal_mat, -1)
#
#             if self.temporal_bias_dim == -1:
#                 batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias
#
#             batch_temporal_mat = tf.expand_dims(batch_temporal_mat, 1)  # (B, 1, T_q, T_v)
#             print("batch_temporal_mat", batch_temporal_mat.shape)
#             scores += batch_temporal_mat
#
#         # 5) 计算注意力权重
#         attn_weights = tf.nn.softmax(scores, axis=-1)  # (B, num_heads, T_q, T_v)
#         attn_weights = self.dropout(attn_weights)
#
#         # 6) 计算输出
#         output = tf.matmul(attn_weights, value)  # (B, num_heads, T_q, d_k)
#         output = tf.transpose(output, perm=[0, 2, 1, 3])  # (B, T_q, num_heads, d_k)
#         output = tf.reshape(output, (batch_size, -1, self.d_model))  # (B, T_q, d_model)
#
#         # 7) 最终投影
#         output = self.proj(output)  # (B, T_q, dim_out)
#         output = self.proj_drop(output)
#         if output_attentions:
#             return output, attn_weights  # 返回输出和注意力权重
#         else:
#             return output

class MHA(Layer):
    def split_heads(self, x, batch_size):
        """
        将最后一个维度分割成 (num_heads, depth).
        转置为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __init__(self, num_heads=8,d_model=512, attn_drop=0.0, proj_drop=0.0,
                  add_temporal_bias=True,
                 temporal_bias_dim=64, use_mins_interval=False,batch_size = 128):
        super(MHA, self).__init__()
        assert d_model % num_heads == 0

        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.scale = self.d_k ** -0.5  # 1/sqrt(dk)
        self.add_temporal_bias = add_temporal_bias
        self.temporal_bias_dim = temporal_bias_dim
        self.use_mins_interval = use_mins_interval
        # 创建三个线性层 (相当于 PyTorch 中的 nn.Linear)
        self.query_layer = tf.keras.layers.Dense(units=d_model, input_shape=(None,d_model),use_bias=False)
        self.key_layer = tf.keras.layers.Dense(units=d_model, input_shape=(None,d_model), use_bias=False)
        self.value_layer = tf.keras.layers.Dense(units=d_model, input_shape=(None,d_model), use_bias=False)
        # Dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=attn_drop)

        # 最终投影层
        self.proj = tf.keras.layers.Dense(units=d_model,input_shape=(None,d_model),use_bias=False)
        self.proj_drop = tf.keras.layers.Dropout(rate=proj_drop)
        if self.add_temporal_bias:
            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                self.temporal_mat_bias_1 = tf.keras.layers.Dense(self.temporal_bias_dim, use_bias=False)
                self.temporal_mat_bias_2 = tf.keras.layers.Dense(1, use_bias=False)
            # elif self.temporal_bias_dim == -1:
            #     self.temporal_mat_bias = nn.Parameter(torch.Tensor(1, 1))
            #     nn.init.xavier_uniform_(self.temporal_mat_bias)


    def call(self, query_input,key_input, output_attentions=False, batch_temporal_mat=None):
        batch_size = tf.shape(query_input)[0]
        # 1) 线性变换
        query = self.query_layer(query_input)  # (B, T_q, d_model)
        key = self.key_layer(key_input)  # (B, T_v, d_model)
        value = self.value_layer(key_input)  # (B, T_v, d_model)
        # 2) 分头
        query = self.split_heads(query, batch_size)  # (B, num_heads, T_q, d_k)
        key = self.split_heads(key, batch_size)  # (B, num_heads, T_v, d_k)
        value = self.split_heads(value, batch_size)  # (B, num_heads, T_v, d_k)

        # 3) 计算注意力得分
        scores = tf.matmul(query, key, transpose_b=True)*self.scale # (B, num_heads, T_q, T_v)
        # scores = tf.matmul(query, key, transpose_b=True) * self.scale  # (B, num_heads, T_q, T_v)
        print("注意力得分形状",scores.shape)

        # # 4) 添加时间偏置（如果需要）
        if self.add_temporal_bias and batch_temporal_mat is not None:
            if self.use_mins_interval:
                batch_temporal_mat = 1.0 / tf.math.log(
                    tf.math.exp(1.0) + (batch_temporal_mat / 60.0))
            else:
                batch_temporal_mat = 1.0 / tf.math.log(tf.math.exp(1.0) + batch_temporal_mat)

            if self.temporal_bias_dim != 0 and self.temporal_bias_dim != -1:
                batch_temporal_mat = self.temporal_mat_bias_2(
                    tf.nn.leaky_relu(
                        self.temporal_mat_bias_1(tf.expand_dims(batch_temporal_mat, -1)), alpha=0.2
                    )
                )
                batch_temporal_mat = tf.squeeze(batch_temporal_mat, -1)

            if self.temporal_bias_dim == -1:
                batch_temporal_mat = batch_temporal_mat * self.temporal_mat_bias

            batch_temporal_mat = tf.expand_dims(batch_temporal_mat, 1)  # (B, 1, T_q, T_v)
            print("batch_temporal_mat", batch_temporal_mat.shape)
            scores += batch_temporal_mat

        # 5) 计算注意力权重
        attn_weights = tf.nn.softmax(scores, axis=-1)  # (B, num_heads, T_q, T_v)
        attn_weights = self.dropout(attn_weights)

        # 6) 计算输出
        output = tf.matmul(attn_weights, value)  # (B, num_heads, T_q, d_k)
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (B, T_q, num_heads, d_k)
        output = tf.reshape(output, (batch_size, -1, self.d_model))  # (B, T_q, d_model)

        # 7) 最终投影
        output = self.proj(output)  # (B, T_q, dim_out)
        output = self.proj_drop(output)
        if output_attentions:
            return output, attn_weights  # 返回输出和注意力权重
        else:
            return output




class LearnableFourierPositionalEncoding:
    def __init__(self, M, F_dim, D, gamma=1.0,names = 'LearnableFourierPositionalEncoding'):

        """
        Initialize the LearnableFourierPositionalEncoding model.

        :param M: Each point has an M-dimensional positional value
        :param F_dim: Depth of the Fourier feature dimension
        :param D: Positional encoding dimension
        :param gamma: Parameter to initialize Wr
        """
        # Store parameters
        self.M = M
        self.F_dim = F_dim
        self.D = D
        self.gamma = gamma

        # Define the input layer
        inputs = Input(shape=(None, M))

        # Define the projection layer (equivalent to a Dense layer without bias)
        Wr = Dense(self.F_dim // 2, use_bias=False)

        # Define the MLP as a Sequential model
        mlp = tf.keras.Sequential([
            Dense(D, activation='gelu'),
            Dense(D)
        ])

        # Define the custom Fourier positional encoding layer using Lambda
        def positional_encoding_layer(x):
            projected = Wr(x)  # Linear transformation
            cosines = tf.math.cos(projected)
            sines = tf.math.sin(projected)
            F = 1 / tf.math.sqrt(tf.cast(self.F_dim, tf.float32)) * tf.concat([cosines, sines], axis=-1)
            Y = mlp(F)
            # pos_enc = Y.reshape((B, N, self.D))
            return Y

        # Apply the custom layer using Lambda
        outputs = positional_encoding_layer(inputs)

        # Build the model
        self.model = Model(inputs=inputs, outputs=outputs, name=names)
        # self.model.summary()


# def summary(self):
    #     """Print the summary of the model."""
    #     self.model.summary()

#

class StackedGRU():
    """
    Handles the Encoder part of the overall encoder-decoder model. The Encoder
    takes as input the trajectory after the embedding is done and outputs the
    hidden state of each layer as well as the feature vector representation of
    each trajectory after being passed through the GRU layers contained within.
    """

    def __init__(self, embedding_size, gru_cell_size, num_gru_layers,
                 gru_dropout_ratio, bidirectional, encoder_h0):
        """
        Initializes the model.

        Inputs:
            input: Shape is (batch_size, in_traj_len, embedding_size) This is
                   the trajectory after each of its spatiotemporal cells have
                   been embedded. in_traj_len depends on the input tensor
        Outputs
            hn: Shape is (batch_size,num_gru_layers * num_directions,
                gru_cell_size). This is the hidden state of each of the GRU
                layers
            output: Shape is (batch_size, in_traj_len, gru_cell_size *
                    num_directions). This is the encoded input trajectory.
        Where:
            - batch_size: Size of input batch
            - in_traj_len: Length of the input trajectory
            - embedding_size: The size of the embedding
            - num_gru_layers: The number of GRU layers
            - num_directions: This is 1 if bidirectional is False. 2 if True.
            - gru_cell_size: The size of the GRU cells

        Arguments:
            embedding_size: (integer) The size of the spatiotemporal cells embedding
            gru_cell_size: (integer) Size of the gru cels
            num_gru_layers: (integer) How many GRus to use
            gru_dropout_ratio: (float) The dropout ratio for the GRUs
            bidirectional: (boolean) Whether or not to use bidirectional GRUs
            encoder_h0: (tensor) Initial state of the GRU
        """
        # (batch_size, in_traj_len, embedding_size)
        # in_traj_len is not static
        inputs = Input((None, embedding_size))

        hn = []
        if bidirectional:
            gru = Bidirectional(GRU(gru_cell_size,
                                    dropout = gru_dropout_ratio,
                                    return_sequences = True))\
                               (inputs, initial_state = encoder_h0)
            hn.append(gru)
            for i in range(num_gru_layers-1):
                gru = Bidirectional(GRU(gru_cell_size,
                                        return_sequences = True,
                                        dropout = gru_dropout_ratio))(gru)
                hn.append(gru)
        else:
            gru = GRU(gru_cell_size, return_sequences = True,
                      dropout = gru_dropout_ratio)\
                      (inputs, initial_state=encoder_h0)
            hn.append(gru)
            for i in range(num_gru_layers-1):
                gru = GRU(gru_cell_size, return_sequences = True,
                          dropout = gru_dropout_ratio)(gru)
                hn.append(gru)
        model = Model(inputs = inputs, outputs = gru,name='GRU')
        self.model = model


class Encoder():
    """
    Handles the Encoder part of the overall encoder-decoder model. The Encoder
    takes as input the trajectory after the embedding is done and outputs the
    hidden state of each layer as well as the feature vector representation of
    each trajectory after being passed through the GRU layers contained within.
    """

    def __init__(self, in_traj_len, embedding, gru):
        """
        Initializes the model.

        Args:
            in_traj_len: (integer) Length of trajectory
            embedding: (keras model) The embedding layer
            gru: (keras model) The stacked GRU model
        """
        inputs = Input(shape=(None, ))  # 假设 M 是输入的最后一个维度
        # print(inputs.shape)
        # positional_encoding = LearnableFourierPositionalEncoding(M=1, F_dim=256, D=256)
        # encoded_positions = positional_encoding.model(inputs)
        # print(encoded_positions.shape)
        embedded = embedding(inputs)
        # print(embedded.shape)
        outputs = gru(embedded)
        model = Model(inputs = inputs, outputs = outputs)
        self.model = model
        # self.model.summary()


class query_Encoder():
    """
    Handles the Encoder part of the overall encoder-decoder model. The Encoder
    takes as input the trajectory after the embedding is done and outputs the
    hidden state of each layer as well as the feature vector representation of
    each trajectory after being passed through the GRU layers contained within.
    """

    def __init__(self, in_traj_len, embedding, gru):
        """
        Initializes the model.

        Args:
            in_traj_len: (integer) Length of trajectory
            embedding: (keras model) The embedding layer
            gru: (keras model) The stacked GRU model
        """
        inputs_1 = Input((None, ),name='query')
        inputs_2 = Input((None, 2 ),name='spa')
        # inputs_3 = Input((None, ),name='longi')
        inputs_3 = Input((None, ),name='tem')
        print("inputs_1: ",inputs_1.shape)
        print("inputs_2: ",inputs_2.shape)
        # print("inputs_3: ",inputs_3.shape)
        # print("inputs_4: ",inputs_4.shape)
        # lat_encoding = LearnableFourierPositionalEncoding(M=1, F_dim=256, D=256,names = 'learnable1')
        spa_encoding = LearnableFourierPositionalEncoding(M=2, F_dim=256, D=256,names = 'learnable2')
        temporal_encoding = LearnableFourierPositionalEncoding(M=1, F_dim=256, D=256, names = 'learnable3')
        encoded_spa = spa_encoding.model(inputs_2)
        # encoded_lon = lon_encoding.model(inputs_3)
        encoded_temporal = temporal_encoding.model(inputs_3)
        embeddings = embedding(inputs_1)
        # print("encoded_positions: ",encoded_positions.shape)
        print("encoded_temporal: ",encoded_temporal.shape)
        print("embeddings: ",embeddings.shape)
        enc = encoded_spa + encoded_temporal + embeddings
        print("enc: ",enc.shape)
        # self.attention = MHA(num_heads=8)
        # embedded = embedding(encoded_positions)
        # print(embedded.shape)
        outputs = gru(enc)
        # outputs = self.attention(outputs)
        model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
        self.model = model
        # self.model.summary()
class Decoder():
    """
    Handles the Decoder part of the overall encoder-decoder model. The Encoder
    first produces the feature vector representation of the target
    trajectory. Then, an attention layer is applied to this feature vector
    representation and the source trajectory feature vector representation.
    The output of this attention module is used as the Decoder
    output.
    """

    def __init__(self, src_traj_len, src_feature_size, trg_traj_len,
                  embedding, gru,use_attention=True):
        """
        Initialize the model

        Inputs:
            inputs_1: Shape is (batch_size, src_traj_len, src_feature_size).
                      This is the learned representation of the source
                      trajectory, which is the output from the Encoder
            inputs_2: Shape is (batch_size, trg_traj_len). This is the target
                      trajectory.

        Outputs:
            outputs: Shape is (batch_size, trg_traj_len, src_feature_size)

        Args:
            src_traj_len: (integer) The length of the source trajectory
            src_feature_size: (integer) Feature vector size of each sequence in
                               the target trajectory.
            trg_traj_len: (integer) The length of the target trajectory
            use_attention: (boolean) Whether or not to use the attention module
            embedding: (keras model) The embedding layer
            gru: (keras model) The GRU model
        """
        # embedded_src (batch_size, src_traj_len, src_feature_size)
        # This input is for the encoded source trajectory
        # self.attention = MHA(num_heads=8, d_model=src_feature_size, src_traj_len=src_traj_len,trg_traj_len=trg_traj_len,dim_out=src_feature_size)

        embedded_src = Input((src_traj_len, src_feature_size))

        # inputs_trg (batch_size, trg_traj_len)
        # This input is for the target trajectory
        inputs_trg = Input((trg_traj_len,))
        batch_temporal_mat = Input((src_traj_len,))
        # time_matrix = Input((src_traj_len,))

        # inputs_trg (batch_size, trg_traj_len)
        # embedded_trg (batch_size, trg_traj_len, src_feature_size)
        # Since the embedding layer used is the same as the target, the
        # output feature size is the same as src_feature_size
        embedded_trg = embedding(inputs_trg)
        embedded_trg = gru(embedded_trg)

        # embedded_src (batch_size, src_traj_len, src_feature_size)
        # embedded_trg (batch_size, trg_traj_len, src_feature_size)
        # outputs (batch_size, trg_traj_len, src_feature_size)
        # TODO: OPTION TO NOT USE ATTENTION
        print("embedded_trg", embedded_trg.shape)
        print("embedded_src", embedded_src.shape)

        outputs = Attention()([embedded_trg, embedded_src])
        #self.attention = MHA(num_heads=8,d_model=src_feature_size)
        #outputs = self.attention(query_input=embedded_trg, key_input=embedded_src,training=True)
        self.model = Model(inputs = [embedded_src, inputs_trg,batch_temporal_mat],outputs=outputs)
        self.model.summary()


class PatternDecoder():
    """
    The pattern decoder. It first takes the target pattern and produces a
    feature vector representation so that the dimension matches the source
    input, which is the source trajectory feature vector representation. Then,
    the two are fed into an attention layer to produce the prediction
    """
    def __init__(self, src_traj_len, src_feature_size, trg_traj_len,use_attention=True
                ):
        """
        Initialize the model

        Inputs:
            inputs_1: Shape is (batch_size, src_traj_len, src_feature_size).
                      This is the learned representation of the source
                      trajectory, which is the output from the Encoder
            inputs_2: Shape is (batch_size, trg_traj_len, 2). This is the
                      target trajectory pattern features, which includes the
                      spatial and temporal features.

        Outputs:
            outputs: Shape is (batch_size, trg_traj_len, src_feature_size)

        Args:
            src_traj_len: (integer) The length of the source trajectory
            src_feature_size: (integer) Feature vector size of each sequence in
                               the target trajectory.
            trg_traj_len: (integer) The length of the target trajectory
            use_attention: (boolean) Whether or not to use the attention module
        """
        # embedded_src (batch_size, src_traj_len, src_feature_size)
        # This input is for the encoded source trajectory
        # self.attention = MHA(num_heads=8, d_model=src_feature_size, dim_out=src_feature_size)
        embedded_src = Input((src_traj_len, src_feature_size))

        # inputs_trg (batch_size, trg_traj_len, 2)
        # This input is for the target spatiotemporal trajectory pattern
        inputs_trg = Input((trg_traj_len, 2))

        # inputs_trg (batch_size, trg_traj_len, 2)
        # embedded_trg (batch_size, trg_traj_len, src_feature_size)
        # Learns features based on inputs_trg.
        embedded_trg = TimeDistributed(Dense(src_feature_size,
                                             activation = 'relu',use_bias=False))(inputs_trg)

        # embedded_src (batch_size, src_traj_len, src_feature_size)
        # embedded_trg (batch_size, trg_traj_len, src_feature_size)
        # outputs (batch_size, trg_traj_len, src_feature_size)
        # TODO: OPTION TO NOT USE ATTENTION
        # self.attention = MHA(num_heads=8, d_model=src_feature_size)
        # outputs = self.attention(embedded_trg, embedded_src)
        # self.attention = MultiHeadAttention(num_heads=8, key_dim=64, use_bias=False)
        # outputs = self.attention(query=embedded_trg, value=embedded_src)
        # self.attention = MultiHeadAttention(num_heads=8, key_dim=64, use_bias=False)
        # outputs = self.attention(query=embedded_trg, value=embedded_src,training=True)
        outputs = Attention()([embedded_trg,embedded_src])
        # self.attention = MultiHeadAttention(num_heads=8, key_dim=64, use_bias=False)
        # outputs = self.attention(query=embedded_trg, value=embedded_src)
        # outputs = self.attention(embedded_src, embedded_trg)
        print("outputs", outputs.shape)
        # outputs = Multiattention(num_heads=8, output_dim=src_feature_size, key_dim=src_feature_size )(
        #     [embedded_src, embedded_trg])
        self.model = Model(inputs = [embedded_src, inputs_trg],outputs=outputs)
        self.model.summary()


class EncoderDecoder():
    """
    The EncoderDecoder model is the final model to be used, which is an
    amalgamation of the Encoder, Decoder, and other components contained
    within the other classes in this module.
    """

    def __init__(self, src_traj_len, trg_traj_len, trg_patt_len,
                 embed_vocab_size, embedding_size, gru_cell_size,
                 num_gru_layers, gru_dropout_ratio, bidirectional,
                 use_attention,
                 encoder_h0 = None):
        """
        Initializes the model.

        Inputs:
            input: Shape is (batch_size, src_traj_len). This is the input
                   trajectory where each item in the sequence is the
                   spatiotemporal cell IDs
        Outputs:
            output: Shape is (batch_size, trg_traj_len, embedding_size). It
                    contains the feature vector representation of the predicted
                    output trajectory.
        Where:
            - batch_size: Size of input batch
            - src_traj_len: Length of the source (i.e. query) trajectory
            - trg_traj_len: Length of the output (i.e. ground truth) trajectory
            - k: Stands for the top-k nearest neighbors' weights.

        Args:
            src_traj_len: (integer) Length of source trajectory
            trg_traj_len: (integer) Length of target trajectory
            embed_vocab_size: (integer) Size of the embedding layer vocabulary
            embedding_size: (integer) Size of the embedding layer output
            gru_cell_size: (integer) Size of the gru cels
            num_gru_layers: (integer) How many GRUs to use
            gru_dropout_ratio: (float) Dropout ratio for the GRUs
            bidirectional: (boolean) Whether or not bidirectional GRU is used
            use_attention: (boolean) Whether or not the attention model is used
            encoder_h0: (tensor) Initial state of the encoder
        """
        # inputs_1 (batch_size, src_traj_len)
        # This input is for the source trajectory
        inputs_1 = Input((src_traj_len,),name='q')
        print("src_traj shape", inputs_1.shape)

        # inputs_2 (batch_size, trg_traj_len)
        # This input is for the target trajectory
        inputs_2 = Input((trg_traj_len,),name='gt')

        # inputs_3 (batch_size, trg_patt_len)
        # This input is for the target pattern
        inputs_3 = Input((trg_patt_len,2),name='patt')
        inputs_4 = Input((src_traj_len,2),name='spa')
        inputs_5 = Input((src_traj_len, ),name='tem')
        # This is for MHA
        inputs_6 = Input((src_traj_len, ),name='temmatrix')

        # Encoder part
        self.sgru = StackedGRU(embedding_size, gru_cell_size,
                               num_gru_layers, gru_dropout_ratio, bidirectional,
                               encoder_h0).model
        self.embedding = Embedding(embed_vocab_size, embedding_size)
        self.encoder = Encoder(src_traj_len,self.embedding,self.sgru)
        self.q_encoder = query_Encoder(src_traj_len, self.embedding, self.sgru)
        # inputs (batch_size, src_traj_len)
        # encoded (batch_size, src_traj_len, gru_cell_size * directions)
        # directions = 2 if bidirectional, else 1.
        encoded = self.q_encoder.model([inputs_1,inputs_4,inputs_5])

        # Decoder part
        traj_repr_size = gru_cell_size
        if bidirectional:
            traj_repr_size *= 2
        self.decoder = Decoder(src_traj_len, traj_repr_size, trg_traj_len,
                               self.embedding, self.sgru,use_attention)
        # encoded (batch_size, src_traj_len, gru_cell_size * directions)
        # inputs_2 (batch_size, trg_traj_len)
        # output_point (batch_size, trg_traj_len, gru_cell_size * directions)
        output_point = self.decoder.model([encoded, inputs_2,inputs_6])

        # Pattern decoder part
        # Pattern decoder
        self.patt_decoder = PatternDecoder(src_traj_len, traj_repr_size,
                                           trg_patt_len,use_attention)
        # encoded (batch_size, src_traj_len, embedding_size)
        # inputs_2 (batch_size, trg_traj_len)
        # output_patt (batch_size, trg_patt_len, embedding_size)
        output_patt = self.patt_decoder.model([encoded, inputs_3])
        print("out_point", output_point.shape)
        print("output_patt", output_patt.shape)

        # Finished model
        self.model = Model(inputs = [inputs_1, inputs_2, inputs_3, inputs_4, inputs_5,inputs_6],
                           outputs = [output_point, output_patt])
        self.model.summary()


class STSeqModel():
    """This class handles the SpatioTemporal Sequence2Sequence """


    def __init__(self, embed_vocab_size, embedding_size, traj_repr_size,
                 gru_cell_size, num_gru_layers, gru_dropout_ratio,
                 bidirectional, k,use_attention):
        """
        Creates the model

        Args:
            gru_cell_size: (integer) Size of every LSTM in the model
            traj_repr_size: (integer) The size of the trajectory vector
                             representation
            xshape: (numpy array) The size of the input numpy arrays
        """
        # 'inputs' shape (batch_size, num_inner_data, traj_len, 1)
        # num_inner_data should be 7, representing:
        # - ground truth trajectory,
        # - query trajectory
        # - negative trajectory
        # - spatial pattern
        # - temporal pattern
        # - jing du
        # - wei du
        # - shi jian
        self.__NUM_FEATURES = 14
        self.__NUM_INNER_FEATURES = 1
        inputs = Input((self.__NUM_FEATURES, None, self.__NUM_INNER_FEATURES))

        ## Lambda layers to split the inputs.
        # 'gt' shape (batch_size, traj_len, 1).
        # Represents ground truth trajectory.
        gt = Lambda(lambda x:x[:,0,:,0])(inputs)
        gt = Masking(mask_value = 0)(gt)

        # 'q' shape (batch_size, traj_len, 1).
        # Represents the query trajectory.
        q = Lambda(lambda x:x[:,1,:,0])(inputs)
        q = Masking(mask_value = 0)(q)
        print("q.shape", q.shape)

        # 'neg' shape (batch_size, traj_len, 1).
        # Represents the negative trajectory.
        neg = Lambda(lambda x:x[:,2,:,0])(inputs)
        neg = Masking(mask_value = 0)(neg)

        # 'gt_patt_s' shape (batch_size, traj_len, 1).
        gt_patt_s = Lambda(lambda x:x[:,3,:,:])(inputs)
        gt_patt_s = Masking(mask_value = 0)(gt_patt_s)

        # 'gt_patt_t' shape (batch_size, traj_len, 1).
        gt_patt_t = Lambda(lambda x:x[:,4,:,:])(inputs)
        gt_patt_t = Masking(mask_value = 0)(gt_patt_t)
        q_lat =  Lambda(lambda x:x[:,5,:,0])(inputs)
        q_lat = Masking(mask_value = 0)(q_lat)
        print('q_lat.shape',q_lat.shape)
        q_lon = Lambda(lambda x:x[:,6,:,0])(inputs)
        q_lon = Masking(mask_value=0)(q_lon)
        print('q_lon.shape',q_lon.shape)
        q_t = Lambda(lambda x:x[:,7,:,0])(inputs)
        q_t = Masking(mask_value=0)(q_t)
        print('q_t.shape',q_t.shape)
        batch_time_temporal = construct_time_matrix(q_t)
        print("batch_time_temporal",batch_time_temporal.shape)
        gt_lat = Lambda(lambda x:x[:,8,:,0])(inputs)
        gt_lat = Masking(mask_value=0)(gt_lat)
        gt_lon = Lambda(lambda x: x[:, 9, :, 0])(inputs)
        gt_lon = Masking(mask_value=0)(gt_lon)
        gt_t = Lambda(lambda x: x[:, 10, :, 0])(inputs)
        gt_t = Masking(mask_value=0)(gt_t)
        neg_lat = Lambda(lambda x: x[:, 11, :, 0])(inputs)
        neg_lat = Masking(mask_value=0)(neg_lat)
        neg_lon = Lambda(lambda x: x[:, 12, :, 0])(inputs)
        neg_lon = Masking(mask_value=0)(neg_lon)
        neg_t = Lambda(lambda x: x[:, 13, :, 0])(inputs)
        neg_t = Masking(mask_value=0)(neg_t)
        q_lat_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(q_lat)  # 形状变为 (batch_size, None, 1)
        q_lon_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(q_lon)  # 形状变为 (batch_size, None, 1)
        #
        # # 在特征维度 (axis=2) 上合并它们
        q_spa = Concatenate(axis=-1)([q_lat_expanded, q_lon_expanded])  # 合并后的形状是 (batch_size, None, 2)
        gt_lat_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(gt_lat)  # 形状变为 (batch_size, None, 1)
        gt_lon_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(gt_lon)  # 形状变为 (batch_size, None, 1)
        #
        # # 在特征维度 (axis=2) 上合并它们
        gt_spa = Concatenate(axis=-1)([gt_lat_expanded, gt_lon_expanded])  # 合并后的形状是 (batch_size, None, 2)
        neg_lat_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(neg_lat)  # 形状变为 (batch_size, None, 1)
        neg_lon_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(neg_lon)  # 形状变为 (batch_size, None, 1)
        #
        # # 在特征维度 (axis=2) 上合并它们
        neg_spa = Concatenate(axis=-1)([neg_lat_expanded, neg_lon_expanded])  # 合并后的形状是 (batch_size, None, 2)
        # 'gt_patt_st' shape (batch_size, traj_len, 2)
        gt_patt_st = Concatenate(axis=2)([gt_patt_s, gt_patt_t])
        print("q shape: ", q.shape)
        # print("q_spa shape: ", q_spa.shape)
        print("q_t shape: ", q_t.shape)
        print("gt_spa shape: ", gt_spa)
        print("neg_spa shape: ", neg_spa)

        # EncoderDecoder model
        assert gt_patt_s.shape[0] == gt_patt_t.shape[0]
        src_traj_len = q.shape[1]
        trg_traj_len = gt.shape[1]
        trg_patt_len = gt_patt_s.shape[1]
        # st_len = q_t.shape[1]

        # Getting the point-to-point and pattern representation
        # Inputs:
        # 'q' shape (batch_size, traj_len, 1).
        # 'gt' shape (batch_size, traj_len, 1).
        # 'gt_patt_st' shape (batch_size, traj_len, 2)
        # Outputs:
        # 'traj_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
        # 'patt_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
        # directions = 2 if bidirectional, else 1
        encoder_decoder = EncoderDecoder(src_traj_len, trg_traj_len,
                                         trg_patt_len, embed_vocab_size,
                                         embedding_size, gru_cell_size,
                                         num_gru_layers, gru_dropout_ratio,
                                         bidirectional,use_attention)
        [traj_repr, patt_repr] = encoder_decoder.model([q, gt, gt_patt_st,q_spa,q_t,batch_time_temporal])

        # Encoder part
        self.encoder = encoder_decoder.encoder
        self.q_encoder = encoder_decoder.q_encoder

        # Getting the trajectory representation
        # Inputs:
        # 'q' shape (batch_size, traj_len, 1).
        # 'gt' shape (batch_size, traj_len, 1).
        # 'neg' shape (batch_size, traj_len, 1).
        # Outputs:
        # 'enc_q' shape (batch_size, traj_len, gru_cell_size * directions).
        # 'enc_gt' shape (batch_size, traj_len, gru_cell_size * directions).
        # 'enc_neg' shape (batch_size, traj_len, gru_cell_size * directions).
        # directions = 2 if bidirectional, else 1
        # q = Input(shape=(src_traj_len,),dtype='float32')
        # q_spa = Input(shape=(src_traj_len,2),dtype='float32')
        # # q_spa = Masking(mask_value=0)(q_spa)
        # q_t = Input(shape=(src_traj_len,1),dtype='float32')
        # print("q shape: ", q.shape)
        # print("q_spa shape: ", q_spa.shape)
        # print("q_t shape: ", q_t.shape)
        # print("gt shape: ", gt)
        # print("neg shape: ", neg)

        enc_gt = self.q_encoder.model([gt,gt_spa,gt_t])
        enc_neg = self.q_encoder.model([neg,neg_spa,neg_t])
        enc_q = self.q_encoder.model([q, q_spa, q_t])

        # Three loss functions needed, so we need three outputs
        # First is the representation loss which takes the encoder outputs
        # Inputs:
        # 'enc_q' shape (batch_size, traj_len, gru_cell_size * directions).
        # 'enc_gt' shape (batch_size, traj_len, gru_cell_size * directions).
        # 'enc_neg' shape (batch_size, traj_len, gru_cell_size * directions).
        # Outputs:
        # 'out_repr' shape (batch_size, 3, traj_len, gru_cell_size * directions)
        out_repr = K.stack([enc_q, enc_gt, enc_neg], axis=1)

        # Second is the point-to-point loss.
        # Inputs:
        # 'traj_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
        # Outputs:
        # 'out_traj' shape (batch_size, trg_traj_len, k)
        out_traj = TimeDistributed(Dense(k, activation = 'relu'))(traj_repr)

        # Third is the pattern loss
        # Inputs:
        # 'patt_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
        # Outputs:
        # 'out_patt' shape (batch_size, trg_traj_len, 2)
        out_patt = TimeDistributed(Dense(2, activation = 'relu'))(patt_repr)

        # Create model
        model = Model(inputs = inputs, outputs = [out_repr, out_traj, out_patt])
        self.model = model
        self.model.summary()