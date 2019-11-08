"""
do not use probabilities but use directly numbers

"""
import csv
from datetime import date, datetime
import tensorflow as tf

import ephem

import numpy as np
import math

class sazka_building(ephem.Observer):

    def __init__(self, date):
        super.__init__()
        self.lon = '14.4963524'
        self.lat = '50.0986794'
        self.elevation = 234
        self.date = date

class draw_history(object):

    draws = []

    def __init__(self):
        with open('sportka.csv', newline='') as csvfile:
            the_reader = csv.reader(csvfile, delimiter=';')
            is_header = True
            for row in the_reader:
                if not is_header:
                    self.draws.append(draw(row, draw_history=self))

                is_header = False

class draw(object):

    def __init__(self, row, draw_history):
        try:
            print(row)
            self.date = datetime.strptime(row[0], '%d. %m. %Y').date()
            self.week = int(row[2])
            self.week_day = int(row[3])
            self.first = [int(x) for x in row[4:11]]
            self.second = [int(x) for x in row[11:18]]
            self.draw_history = draw_history
            #self.observer = sazka_building(self.date)

            print('>OK')
        except:
            print('>ERROR')

    @property
    def x_train(self):
        return date_to_x(self.date)

    @property
    def y_train_1(self):
        return np.array(self.first)

    @property
    def y_train_2(self):
        return np.array(self.second)

    @property
    def y_train_random(self):
        random_numbers = []
        for _  in range(7):
            while True:
                rn = random(1, 49)
                if rn not in random_numbers:
                    random_numbers.append(rn)
                    break
                else:
                    continue
        return np.array(self.random_numbers)

    @property
    def x_train_history_1(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference
        if history_index >= 0:
            return self.draw_history.draws[history_index].y_train_1
        else:
            return y_train_random

    @property
    def x_train_history_2(self):
        difference = 1
        index = self.draw_history.draws.index(self)
        history_index = index - difference
        if history_index >= 0:
            return self.draw_history.draws[history_index].y_train_2
        else:
            return y_train_random

    @property
    def observer(self):
        return sazka_building


def date_to_x(date):

    #consider also some astrological data
    previous_new_moon = ephem.previous_new_moon(date)
    next_new_moon = ephem.next_new_moon(date)
    relative_lunation = (ephem.Date(date) - previous_new_moon) / (next_new_moon - previous_new_moon)

    return np.array([date.day / 31.0, date.month / 12.0, date.year / 2019.0, date.weekday() / 6.0, relative_lunation])

def normalize(array):
    return np.array([((x-1) % 48 + 1) for x in array])

def loss_function(x_true, y_pred):
    x_duplicite = sum([len([x for x in x_true if x == x_value]) for x_value in x_true])
    y_duplicite = sum([len([y for y in y_pred if y == y_value]) for y_value in y_pred])
    duplicities = x_duplicite + y_duplicite
    matching = sum([1 for y in y_pred if y in x_pred])

    f = math.factorial
    n = len(x_true)
    return math.sqr(f(n + duplicities) / f(matching) / f(n + duplicities-matching) + random.random() - 0.5)

def learn_and_predict_sportka(x_train, y_train_both, x_predict, depth=128, epochs=15):

    inputs = tf.keras.Input(shape=(19,))  # Returns a placeholder tensor

    x = tf.keras.layers.Dense(128, activation='sigmoid',
                              kernel_initializer='random_normal',
                              bias_initializer='random_normal')(inputs)

    for i in range(1, depth - 2):
        x = tf.keras.layers.Dense(128, activation='sigmoid',
                                  kernel_initializer='random_normal',
                                  bias_initializer='random_normal'
                                  )(x)
        x = tf.keras.layers.Dropout(0.4)(x)

    for i in range(1, depth_wide):
        #x = tf.keras.layers.Dense(2450, activation='relu')(x)
        x = tf.keras.layers.Dense(6, activation='sigmoid',
                                  kernel_initializer='random_normal',
                                  bias_initializer='random_normal'
                                  )(x)
        x = tf.keras.layers.Dropout(0.4)(x)

    predictions = normalize(tf.keras.layers.Dense(6, activation='linear')(x))

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.AdamOptimizer(0.0005), loss='loss_function')

    model.fit(x=x_train, y=y_train_both, epochs=epochs)
    return model.predict(x_predict)


########################################################################################################################
############################## main program ############################################################################
########################################################################################################################

DATE_PREDICT = '4.8.2019'

dh = draw_history()
print(dh)

x_predict = np.array([date_to_x(datetime.strptime(DATE_PREDICT, '%d.%m.%Y').date())])
x_predict_draw_1 = np.array([dh.draws[-1].y_train_1])
x_predict_draw_2 = np.array([dh.draws[-1].y_train_2])
x_predict_all = [np.concatenate((x_predict, x_predict_draw_1, x_predict_draw_2), axis=1)]

x_train_all = np.array(
    [np.concatenate((draw.x_train, draw.x_train_history_1, draw.x_train_history_2), axis=0) for draw in dh.draws])

y_train_1 = np.array([draw.y_train_1 for draw in dh.draws])
y_train_2 = np.array([draw.y_train_2 for draw in dh.draws])

y_predict_1 = learn_and_predict_sportka(x_train_all, y_train_1, x_predict_all, depth=128, epochs=500)
y_predict_numbers_1 = y_predict_1[:49]
y_predict_2 = learn_and_predict_sportka(x_train_all, y_train_2, x_predict_all, depth=128, epochs=500)
y_predict_numbers_2 = y_predict_2[:49]

print('first draw ')
print('best numbers for {}\n: {}\n\n'.format(DATE_PREDICT, y_predict_1))

print('second draw {}')
print('best numbers for {}\n: {}\n\n'.format(DATE_PREDICT, y_predict_2))


#print('combined :')
#print('best numbers for {}\n: {}\n\n'.format(DATE_PREDICT, best_numbers(y_predict_numbers_1 + y_predict_numbers_2, 6)))
#print('all numbers\n: {}\n\n'.format(best_numbers(y_predict_numbers_1 + y_predict_numbers_2, 49)))
