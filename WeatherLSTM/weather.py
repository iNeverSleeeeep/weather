import tensorflow as tf
import numpy as np
import os
import codecs

n_weather = None
n_wind = None

def flat(a):
    b = []
    for each in a:
        if not isinstance(each, list):
            b.append(each)
        else:
            b.extend(flat(each))
    return b

def to_one_hot(value, list):
    one_hot = [0 for _ in list]
    one_hot[list.index(value)] = 1
    return one_hot

def from_one_hot(one_hot, list):
    return list[one_hot.index(1)]

def get_weather_data(filename):
    def parse_line(line):
        raw_list = line.split(',')
        list = []
        list.append(raw_list[1])
        list.append(float(raw_list[2]))
        list.append(float(raw_list[3]))
        list.append(raw_list[4])
        list.append(float(raw_list[5]))
        return list
    
    data = []
    f = codecs.open(filename, 'r', 'utf-8')
    while True:
        line = f.readline()
        if line:
            data.append(parse_line(line))
        else:
            break;
    f.close()

    weather = [d[0] for d in data]
    wind = [d[3] for d in data]

    weather_list = list(set(weather))
    wind_list = list(set(wind))

    global n_weather
    global n_wind
    n_weather = len(weather_list)
    n_wind = len(wind_list)

    for day_data in data:
        day_data[0] = to_one_hot(day_data[0], weather_list)
        day_data[3] = to_one_hot(day_data[3], wind_list)

    return data

def get_batches(data, n_seqs, n_steps):
    global n_weather
    global n_wind

    batch_size = n_seqs * n_steps
    n_batches = int(len(data)/batch_size)
    input_size = n_weather+n_wind+3

    data_temp = []
    for d in data:
        data_temp.append(flat(d))

    arr = np.array(data_temp[-batch_size*n_batches:])
    
    arr = arr.reshape((n_seqs, -1, input_size))
    for n in range(0, arr.shape[1]-n_steps, n_steps):
        x = arr[:, n:n+n_steps]
        y = arr[:, n+1:n+n_steps+1]
        y = y.reshape((-1, input_size))
        yy = y[:, n_weather+1]
        yy = yy.reshape(n_seqs, n_steps, 1)
        yield x, yy

def build_inputs(n_seqs, n_steps):
    global n_weather
    global n_wind
    inputs_weather = tf.placeholder(tf.float32, [n_seqs, n_steps, n_weather])
    inputs_high_temperature = tf.placeholder(tf.float32, [n_seqs, n_steps, 1])
    inputs_low_temperature = tf.placeholder(tf.float32, [n_seqs, n_steps, 1])
    inputs_wind = tf.placeholder(tf.float32, [n_seqs, n_steps, n_wind])
    inputs_wind_rank = tf.placeholder(tf.float32, [n_seqs, n_steps, 1])
    
    inputs = tf.concat([inputs_weather, inputs_high_temperature, inputs_low_temperature, inputs_wind, inputs_wind_rank], 2)

    targets_low_temperature = tf.placeholder(tf.float32, [n_seqs, n_steps, 1])

    return inputs, targets_low_temperature

def build_lstm(lstm_size, n_layers, n_seqs):

    list = []
    for _ in range(n_layers):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        list.append(lstm)

    if n_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell(list, state_is_tuple = True)
    else:
        cell = list[0]

    initial_state = cell.zero_state(n_seqs, tf.float32)
    return cell, initial_state

def build_output(lstm_output, lstm_size, out_size):
    print(lstm_output)
    print(lstm_size)
    print(out_size)
    seq_output = tf.concat(lstm_output, 1)
    print(seq_output)

    x = tf.reshape(seq_output, [-1, lstm_size])

    w = tf.Variable(tf.truncated_normal([lstm_size, out_size], stddev=0.1))
    b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, w) + b
    out = logits

    return out, logits

def build_loss(logits, targets, lstm_size, out_size):
    y = tf.reshape(targets, logits.get_shape())

    temp1 = tf.reshape(y, [-1])
    temp2 = tf.reshape(logits, [-1])
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=temp2, labels=temp1)
    loss = tf.square(temp1 - temp2)
    loss = tf.reduce_mean(loss)

    return loss

def build_optimizer(loss, learning_rate, grad_clip):

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer

class WeatherRNN:

    def __init__(self, n_seqs=1, n_steps=1, lstm_size=512, n_layers=1, learning_rate=0.001, grad_clip = 5):

        tf.reset_default_graph()

        # 输入层
        self.inputs, self.targets = build_inputs(n_seqs, n_steps)
        # lstm层
        cell, self.initial_state = build_lstm(lstm_size, n_layers, n_seqs)
        # run
        outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)

        self.final_state = state

        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, 1)

        # loss optimizer
        self.loss = build_loss(self.logits, self.targets, lstm_size, 1)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)



def train():
    n_seqs = 12         # Sequences per batch
    n_steps = 30          # Number of sequence steps per batch
    lstm_size = 512         # Size of hidden layers in LSTMs
    n_layers = 1          # Number of LSTM layers
    learning_rate = 0.0001    # Learning rate

    epochs = 40

    weather_data = get_weather_data("../data/chengdu.csv")
    model = WeatherRNN(n_seqs, n_steps, lstm_size, n_layers, learning_rate)

    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0

        for e in range(epochs):
            new_state = sess.run(model.initial_state)
            loss = 0
            for x, y in get_batches(weather_data, n_seqs, n_steps):
                counter += 1
                feed = {model.inputs:x, model.targets:y, model.initial_state:new_state}

                batch_loss, new_state, _ = sess.run([model.loss, model.final_state, model.optimizer], feed_dict=feed)

                print('轮数: {}/{}... '.format(e+1, epochs),
                        '训练步数: {}... '.format(counter),
                        '训练误差: {:.4f}... '.format(batch_loss))


        saver.save(sess, "checkpoints/weather.ckpt")

def test():
    weather_data = get_weather_data("../data/chengdu.csv")
    model = WeatherRNN()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "checkpoints/weather.ckpt")
        new_state = sess.run(model.initial_state)
        data = weather_data[-90:]
        for x, y in get_batches(data, 1, 1):
            feed = {model.inputs:x, model.targets:y, model.initial_state:new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)
            print(preds)
        

            
        


if __name__ == '__main__':
    #train()
    test()
