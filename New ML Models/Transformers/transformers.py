import tensorflow as tf
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import register_plotly_resampler
import joblib


class TTdata(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size, enc_seq_len, target_seq_len):
        self.X = X
        self.Y = Y
        self.enc_seq_len = enc_seq_len
        self.target_seq_len = target_seq_len
        self.data_len = self.X.shape[0] - self.enc_seq_len - self.target_seq_len
        self.batch_size = batch_size

    def __getitem__(self, batch_idx):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """

        rows = np.arange(batch_idx * self.batch_size, min(self.batch_size * (batch_idx + 1), self.data_len))
        src = np.array([self.X[index:index + self.enc_seq_len] for index in rows])
        trg = np.array(
            [self.Y[index + self.enc_seq_len - 1:index + self.target_seq_len - 1 + self.enc_seq_len] for index in rows])
        trg_y = np.array(
            [self.Y[index + self.enc_seq_len: index + self.enc_seq_len + self.target_seq_len] for index in rows])

        return (src, trg), trg_y

    def __len__(self):
        return math.ceil(self.data_len / self.batch_size)


class Inf_TTdata(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size, enc_seq_len, target_seq_len):
        self.X = X
        self.Y = Y
        self.enc_seq_len = enc_seq_len
        self.target_seq_len = target_seq_len
        self.data_len = self.X.shape[0] - self.enc_seq_len - self.target_seq_len
        self.batch_size = batch_size

    def __getitem__(self, batch_idx):
        """
        Returns a tuple with 2 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        """

        rows = np.arange(batch_idx * self.batch_size, min(self.batch_size * (batch_idx + 1), self.data_len))
        src = np.array([self.X[index:index + self.enc_seq_len] for index in rows])
        trg = np.array(
            [self.Y[index + self.enc_seq_len - 1:index + self.target_seq_len - 1 + self.enc_seq_len] for index in rows])

        return src, trg

    def __len__(self):
        return math.ceil(self.data_len / self.batch_size)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_length=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = tf.keras.layers.Dropout(dropout)
        depth = d_model / 2
        positions = np.arange(max_length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

        angle_rates = 1 / (10000 ** depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "pos_encoding": self.pos_encoding, "dropout": self.dropout})
        return config

    def call(self, x):
        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        # x = self.dropout(x)
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def get_config(self):
        config = super().get_config()
        config.update({"mha": self.mha, "layernorm": self.layernorm, "add": self.add})
        return config


class CrossAttention(BaseAttention):
    def get_config(self):
        config = super(BaseAttention, self).get_config()
        # config.update({"mha": self.mha, "last_attn_scores": self.last_attn_scores, "add": self.add,
        #                "layernorm": self.layernorm})
        config.update({"last_attn_scores": self.last_attn_scores})
        return config

    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def get_config(self):
        config = super(BaseAttention, self).get_config()
        # config.update({"mha": self.mha, "add": self.add, "layernorm": self.layernorm})
        # config = {"mha": self.mha, "add": self.add, "layernorm": self.layernorm}
        return config

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def get_config(self):
        config = super(BaseAttention, self).get_config()
        # config.update({"mha": self.mha, "add": self.add, "layernorm": self.layernorm})
        return config

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super().get_config()
        config.update({"add": self.add, "layer_norm": self.layer_norm})
        return config

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def get_config(self):
        config = super().get_config()
        config.update({"self_attention": self.self_attention, "ffn": self.ffn})
        return config

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
                           for _ in range(num_layers)]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_layers": self.num_layers, "enc_layers": self.enc_layers})
        return config

    def call(self, x):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.last_attn_scores = None
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)

        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def get_config(self):
        config = super().get_config()
        config.update({"causal_self_attention": self.causal_self_attention, "cross_attention": self.cross_attention,
                       "last_attn_scores": self.last_attn_scores, "ffn": self.ffn})
        return config

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
                           for _ in range(num_layers)]

        self.last_attn_scores = None

    def get_config(self):
        config = super().get_config()
        config.update({"num_layers": self.num_layers, "dec_layers": self.dec_layers,
                       "last_attn_scores": self.last_attn_scores})
        return config

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, out_features, dropout_rate=0.1):
        super().__init__()
        self.encoder_input = tf.keras.layers.Dense(d_model)
        self.pos_embedding_enc = PositionalEmbedding(d_model=d_model, dropout=dropout_rate)
        self.pos_embedding_dec = PositionalEmbedding(d_model=d_model, dropout=dropout_rate)

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               dropout_rate=dropout_rate)

        self.decoder_input = tf.keras.layers.Dense(d_model)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(out_features)

        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config.update({"encoder_input": self.encoder_input, "pos_embedding": self.pos_embedding,
                       "encoder": self.encoder, "decoder_input": self.decoder_input,
                       "decoder": self.decoder, "final_layer": self.final_layer})
        return config

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder_input(context)
        context = self.pos_embedding_enc(context)
        context = self.dropout_1(context)
        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder_input(x)
        x = self.pos_embedding_dec(x)
        x = self.dropout_2(x)
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        config = {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}
        return config

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def read_transform(csv_path):
    df = pd.read_csv(csv_path)
    nh = df.pop("NH")
    nl = df.pop("NL")
    df.insert(len(df.columns), "NH", nh)
    df.insert(len(df.columns), "NL", nl)
    df = df.fillna(0)
    col_order = df.columns
    return df, col_order.to_list()


def train_transformers(train_data, val_data, train_test_ratio=0.8, epochs=30, num_layers=4, d_model=128, dff=512,
                       num_heads=8, dropout_rate=0.1, input_length=4, output_length=1, batch_size=16, es_patience=4,
                       opt_beta_1=0.9, opt_beta_2=0.98, opt_epsilon=1e-9):
    scaler = MinMaxScaler()
    train_data[:] = scaler.fit_transform(train_data)
    val_data[:] = scaler.transform(val_data)

    X_train, Y_train = train_data, train_data.drop(['NH', 'NL'], axis=1)
    X_val, Y_val = val_data, val_data.drop(['NH', 'NL'], axis=1)

    train_batches = TTdata(X_train, Y_train, batch_size, input_length, output_length)
    val_batches = TTdata(X_val, Y_val, batch_size, input_length, output_length)

    learning_rate = CustomSchedule(d_model)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=opt_beta_1, beta_2=opt_beta_2, epsilon=opt_epsilon)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience)
    out_features = Y_val.shape[1]

    model = Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        out_features=out_features,
                        dropout_rate=dropout_rate)

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer,
                  metrics=[tf.metrics.MeanAbsoluteError()])

    model.fit(train_batches, epochs=epochs, validation_data=val_batches, callbacks=[callback])

    print(model.summary())

    return model, scaler


def test_transformers(model, data, scaler, failed_sens: list, failed_idx: list, batch_size=16, input_length: int = 4,
                      output_length: int = 1):
    data_col_order = data.columns
    data_ = data.copy()
    data_[:] = scaler.transform(data_)

    for idx in range(len(failed_sens)):
        data_.loc[failed_idx[idx]:, failed_sens[idx]] = 0

    data_speed_drop = data_.copy()
    data_speed_drop.drop(data_[["NH", "NL"]], axis=1, inplace=True)

    data_batches = Inf_TTdata(data_, data_speed_drop, batch_size, input_length, output_length)

    itr = True
    for i in tqdm(range(len(data_batches))):
        pred = model.predict(data_batches.__getitem__(i), verbose=0)
        if not itr:
            pred_array = np.append(pred_array, pred, axis=0)
        else:
            pred_array = pred
            itr = False
        del pred

    pred_array = pred_array.reshape((pred_array.shape[0], pred_array.shape[2]))

    pred_df = pd.DataFrame(pred_array, columns=data_col_order[:-2])
    # pred_df = pd.concat([pred_df, data_[["NH", "NL"]][input_length + 1:].reset_index(drop=True)], axis=1)

    virtual_sens_df = data_.copy()

    for idx in range(len(failed_sens)):
        virtual_sens_df.loc[failed_idx[idx]:, failed_sens[idx]] = pred_df.loc[failed_idx[idx] - (input_length + 1):,
                                                                  failed_sens[idx]].to_numpy()

    virtual_sens_df[:] = scaler.inverse_transform(virtual_sens_df)

    mae, mse, rmse, r2 = [], [], [], []
    for idx in range(len(failed_sens)):
        y_true = data.loc[failed_idx[idx]:, failed_sens[idx]]
        y_pred = virtual_sens_df.loc[failed_idx[idx]:, failed_sens[idx]]

        mae.append(mean_absolute_error(y_true, y_pred))
        mse.append(mean_squared_error(y_true, y_pred))
        rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2.append(r2_score(y_true, y_pred))

    test_metrics = pd.DataFrame({"Sensor Name": failed_sens, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2_squared": r2})

    pred_df = pd.concat([pred_df, data_[["NH", "NL"]][input_length + 1:].reset_index(drop=True)], axis=1)
    pred_df[:] = scaler.inverse_transform(pred_df)

    return virtual_sens_df, test_metrics, pred_df


def test_transformers_2(model, data, scaler, failed_sens: list, failed_idx: list, batch_size=16, input_length: int = 4,
                        output_length: int = 1):
    data_col_order = data.columns.to_list()
    failed_col_id = [data_col_order.index(x) for x in failed_sens]
    data_ = data.copy()
    data_[:] = scaler.transform(data_)

    for idx in range(len(failed_sens)):
        data_.loc[failed_idx[idx]:, failed_sens[idx]] = 0

    data_speed_drop = data_.copy()
    data_speed_drop.drop(data_[["NH", "NL"]], axis=1, inplace=True)

    enc_ip = np.array(data_.loc[0:input_length - 1]).reshape([1, input_length, data_.shape[1]])
    dec_ip = np.array(data_speed_drop.loc[input_length - 1, :]).reshape([1, output_length, data_speed_drop.shape[1]])

    itr = True
    idx = 1
    for i in tqdm(range(len(data_) - input_length)):
        pred = model.predict((enc_ip, dec_ip), verbose=0)

        if not itr:
            pred_array = np.append(pred_array, pred, axis=0)
        else:
            pred_array = pred
            itr = False

        enc_ip_ = data_.iloc[idx + input_length - 1, :].copy()
        enc_ip_[failed_sens] = pred[-1, -1, failed_col_id]
        enc_ip = np.append(enc_ip[:, -(input_length - 1):, :], (np.array(enc_ip_))).reshape(1, input_length,
                                                                                            data_.shape[1])
        dec_ip = np.array(enc_ip_)[:-2].reshape([1, output_length, data_speed_drop.shape[1]])

        del pred
        idx = idx + 1

    pred_array = pred_array.reshape((pred_array.shape[0], pred_array.shape[2]))

    pred_df = pd.DataFrame(pred_array, columns=data_col_order[:-2])

    virtual_sens_df = data_.copy()

    for idx in range(len(failed_sens)):
        virtual_sens_df.loc[failed_idx[idx]:, failed_sens[idx]] = pred_df.loc[failed_idx[idx] - (input_length):,
                                                                  failed_sens[idx]].to_numpy()

    virtual_sens_df[:] = scaler.inverse_transform(virtual_sens_df)

    mae, mse, rmse, r2 = [], [], [], []
    for idx in range(len(failed_sens)):
        y_true = data.loc[failed_idx[idx]:, failed_sens[idx]]
        y_pred = virtual_sens_df.loc[failed_idx[idx]:, failed_sens[idx]]

        mae.append(mean_absolute_error(y_true, y_pred))
        mse.append(mean_squared_error(y_true, y_pred))
        rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2.append(r2_score(y_true, y_pred))

    test_metrics = pd.DataFrame({"Sensor Name": failed_sens, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2_squared": r2})

    pred_df = pd.concat([pred_df, data_[["NH", "NL"]][input_length:].reset_index(drop=True)], axis=1)
    pred_df[:] = scaler.inverse_transform(pred_df)

    return virtual_sens_df, test_metrics, pred_df


def inference_transformers(model, data, scaler, failed_sens: list, failed_idx: list, batch_size=16,
                           input_length: int = 4,
                           output_length: int = 1):
    data_col_order = data.columns
    data_ = data.copy()
    data_[:] = scaler.transform(data_)

    for idx in range(len(failed_sens)):
        data_.loc[failed_idx[idx]:, failed_sens[idx]] = 0

    data_speed_drop = data_.copy()
    data_speed_drop.drop(data_[["NH", "NL"]], axis=1, inplace=True)

    data_batches = Inf_TTdata(data_, data_speed_drop, batch_size, input_length, output_length)

    itr = True
    for i in tqdm(range(len(data_batches))):
        pred = model.predict(data_batches.__getitem__(i), verbose=0)
        if not itr:
            pred_array = np.append(pred_array, pred, axis=0)
        else:
            pred_array = pred
            itr = False
        del pred

    pred_array = pred_array.reshape((pred_array.shape[0], pred_array.shape[2]))

    pred_df = pd.DataFrame(pred_array, columns=data_col_order[:-2])
    pred_df = pd.concat([pred_df, data_[["NH", "NL"]][input_length + 1:].reset_index(drop=True)], axis=1)

    virtual_sens_df = data_.copy()

    for idx in range(len(failed_sens)):
        virtual_sens_df.loc[failed_idx[idx]:, failed_sens[idx]] = pred_df.loc[failed_idx[idx] - (input_length + 1):,
                                                                  failed_sens[idx]].to_numpy()

    virtual_sens_df[:] = scaler.inverse_transform(virtual_sens_df)

    return virtual_sens_df


def plot(actual_data, pred_data, nh, nl, sens_name):
    register_plotly_resampler(mode='auto', default_n_shown_samples=10000)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    time = nh.index / 10

    fig.add_trace(go.Scatter(x=time, y=nh, mode='lines', name="NH"), secondary_y=True)
    fig.add_trace(go.Scatter(x=time, y=nl, mode='lines', name="NL"), secondary_y=True)
    fig.add_trace(go.Scatter(x=time, y=actual_data, mode='lines', name=sens_name + "_act"), secondary_y=False)
    fig.add_trace(go.Scatter(x=time, y=pred_data, mode='lines', name=sens_name + "_pred"), secondary_y=False)
    fig.update_layout(title_text="Transformers", height=600)
    # Naming axes
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="NH & NL (%)", secondary_y=True)
    fig.update_yaxes(title_text="Actual & Prediction", secondary_y=False)
    fig.show_dash(host=f"127.0.0.4", port=f"8001")


def overall_metrics(act_df, pred_df):
    mae, mse, rmse, r2 = [], [], [], []
    for sens in act_df.columns:
        y_true = act_df[sens][-pred_df.shape[0]:]
        y_pred = pred_df[sens]

        mae.append(mean_absolute_error(y_true, y_pred))
        mse.append(mean_squared_error(y_true, y_pred))
        rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2.append(r2_score(y_true, y_pred))

    metrics = pd.DataFrame(
        {"Sensor Name": test_data.columns, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2_squared": r2})
    return metrics


if __name__ == "__main__":

    # Train the model
    train_csv_path = r'C:\Users\USER\Desktop\Work\Virtual Sensor Enhancement\New ML models\test_data\HF_train_data_1.csv'
    val_csv_path = r"C:\Users\USER\Desktop\Work\Virtual Sensor Enhancement\New ML models\test_data\HF_inf_data_1.csv"

    train_test_ratio = 0.8
    epochs = 30
    num_layers = 4  # 6
    d_model = 128  # 512
    dff = 512  # 2048
    num_heads = 21
    dropout_rate = 0.1
    input_length = 32
    output_length = 1
    batch_size = 16
    es_patience = 6
    opt_beta_1 = 0.9
    opt_beta_2 = 0.98
    opt_epsilon = 1e-9

    train_data, col_order = read_transform(train_csv_path)
    val_data, val_col_order = read_transform(val_csv_path)

    model, scaler = train_transformers(train_data=train_data, val_data=val_data,
                                       train_test_ratio=train_test_ratio,
                                       epochs=epochs,
                                       num_layers=num_layers,
                                       d_model=d_model,
                                       dff=dff,
                                       num_heads=num_heads,
                                       dropout_rate=dropout_rate,
                                       input_length=input_length,
                                       output_length=output_length,
                                       batch_size=batch_size,
                                       es_patience=es_patience,
                                       opt_beta_1=opt_beta_1,
                                       opt_beta_2=opt_beta_2,
                                       opt_epsilon=opt_epsilon)

    # Testing the model
    failed_sens = ["Sensor A"]
    failed_idx = [50]

    test_data_csv_path = r"C:\Users\USER\Desktop\Work\Virtual Sensor Enhancement\New ML models\test_data\HF_test_data_1.csv"
    test_data, test_col_order = read_transform(test_data_csv_path)

    virtual_sens_df, test_metrics, pred_only_df = test_transformers(model=model,
                                                                    data=test_data,
                                                                    scaler=scaler,
                                                                    failed_sens=failed_sens,
                                                                    failed_idx=failed_idx,
                                                                    batch_size=batch_size,
                                                                    input_length=input_length,
                                                                    output_length=output_length)

    print("\n", virtual_sens_df.head())
    print("\n", test_metrics)

    metrics = overall_metrics(test_data, pred_only_df)
    print("\n", metrics)

    print("<--------- test 2 --------->")

    virtual_sens_df_2, test_metrics_2, pred_only_df_2 = test_transformers_2(model=model,
                                                                            data=test_data,
                                                                            scaler=scaler,
                                                                            failed_sens=failed_sens,
                                                                            failed_idx=failed_idx,
                                                                            batch_size=batch_size,
                                                                            input_length=input_length,
                                                                            output_length=output_length)

    print("\n", virtual_sens_df_2.head())
    print("\n", test_metrics_2)

    metrics_2 = overall_metrics(test_data, pred_only_df_2)
    print("\n", metrics_2)

    # Plotting
    for failed_sen in failed_sens:
        plot(actual_data=test_data[failed_sen], pred_data=virtual_sens_df[failed_sen], nh=test_data["NH"],
             nl=test_data["NL"], sens_name=failed_sen)
