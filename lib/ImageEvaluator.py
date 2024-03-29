import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import pickle

# I have trained the model using the MS-COCO dat-set with reference to the professor's link in the syllabus
# I saved the values that were generated after the training was done
# I then proceeded to reverse engineer the code that the professor had referenced to caption a new image on run time
# I loaded all the data that was generated during training and processed a new image to create captions
# Below is the code


tokenizer = None
meta_dict = None
encoder = None
decoder = None
image_features_extract_model = None
max_length = 0
attention_features_shape = 0
embedding_dim = 0
units = 0
vocab_size = 0


def load_all_data():

    # with open('E:/Data Mining/weights/tokenizer.gru.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)

    with open('/home/recklessPaul94/tokenizer.gru.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # with open('E:/Data Mining/weights/meta.gru.pickle', 'rb') as handle:
    #     meta_dict = pickle.load(handle)

    with open('/home/recklessPaul94/meta.gru.pickle', 'rb') as handle:
        meta_dict = pickle.load(handle)
    #
    max_length = meta_dict['max_length']

    attention_features_shape = meta_dict['attention_features_shape']

    embedding_dim = meta_dict['embedding_dim']

    units = meta_dict['units']

    vocab_size = meta_dict['vocab_size']
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # encoder.load_weights('E:/Data Mining/weights/encoder.gru')
    # decoder.load_weights('E:/Data Mining/weights/decoder.gru')
    # decoder.attention.load_weights('E:/Data Mining/weights/attention.gru')
    encoder.load_weights('/home/recklessPaul94/encoder.gru')
    decoder.load_weights('/home/recklessPaul94/decoder.gru')
    decoder.attention.load_weights('/home/recklessPaul94/attention.gru')


def gru(units):
    # attends over the image to predict the next word
    return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')


class BahdanauAttention(tf.keras.Model):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.units)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


def load_image(image_path):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

# code to test new images
test = []

image_url = "https://images.craigslist.org/00u0u_24XW8uhLYPb_600x450.jpg"
if not image_url:
    test.append(' '.join('NA'))

if type(image_url) != str:
    test.append(' '.join('NA'))

image_url_final = str(image_url).strip()
if len(image_url_final) < 5:
    test.append(' '.join('NA'))

image_extension = image_url_final[-4:]
if "jpg" not in image_extension:
    test.append(' '.join('NA'))
try:
    print("10th")
    image_path = tf.keras.utils.get_file(('image7') + image_extension,
                                         origin=image_url_final)
    result, attention_plot = evaluate(image_path)
    test.append(' '.join(result))
except Exception as e:
    print(e)
    test.append(' '.join('NA'))

print(test)
