import csv
from datetime import date, datetime
import tensorflow as tf
import ephem

import numpy as np
#import keras
#from keras.utils.vis_utils import plot_model


sazka_building = ephem.Observer()

class sazka_building(ephem.Observer()):
    def __init__(date):
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
                if is_header==False:
                    #print(row)
                    self.draws.append(draw(row))
                is_header=False

                
class draw(object):
    
    def __init__(self, row):
        try:
            print(row)
            self.date = datetime.strptime(row[0] + '20:00', '%d. %m. %Y %h:%m').date()
            self.week = int(row[2])
            self.week_day = int(row[3])
            self.first = [int(x) for x in row[4:11]]
            self.second = [int(x) for x in row[11:18]]
            self.observer = sazka_building(self.date)



            print('>OK')
        except:
            print('>ERROR')

    @property
    def x_train(self):
        return date_to_x(self.date)

    @property
    def y_train(self):
        probability_first = np.array([1.0 if  number in self.first else 0 for number in range(1, 50)])
        probability_second = np.array([1.0 if  number in self.first else 0 for number in range(1, 50)])
        return 0.5*(probability_first + probability_second)

    @property
    def observer(self):
        return sazka_building

def date_to_x(date):

    #consider also some astrological data
    previous_new_moon = ephem.previous_new_moon(date)
    next_new_moon = ephem.next_new_moon(date)
    relative_lunation = (ephem.Date(date) - previous_new_moon) / (next_new_moon - previous_new_moon)

    return np.array([
        date.day / 31.0,
        date.month / 12.0,
        date.year / 2019.0,
        date.weekday() / 6.0,
        relative_lunation
    ])

def learn_tutorial():

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

def learn_and_predict_sportka(x_train, y_train, x_predict):

    model = tf.keras.models.Sequential([
        #tf.keras.layers.Flatten(input_shape=(4,)),
        tf.keras.layers.Dense(512, input_shape=(5,), activation=tf.nn.relu),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(49, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    #labels = tf.keras.utils.to_categorical(y_train, num_classes=49)
    model.fit(x=x_train, y=y_train, epochs=10)
    #model.evaluate(x_test, y_test)

    return model.predict(x_predict)


def learn_and_predict_sportka2(x_train, y_train, x_predict):

    inputs = tf.keras.Input(shape=(5,))  # Returns a placeholder tensor

    x = tf.keras.layers.Dense(512, activation='relu')(inputs)

    x = tf.keras.layers.Dense(256, activation='relu')(x)

    predictions = tf.keras.layers.Dense(49, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, epochs=1000)

    return model.predict(x_predict)


def best_numbers(y_predict, n=6):
    numbers_vs_chances = ((i+1, y_predict[0][i]) for i in range(49))
    sorted_numbers = sorted(numbers_vs_chances, key=lambda x: x[1], reverse=True)
    return [key for key in sorted_numbers[0: n]]

########################################################################################################################
############################## main program ############################################################################
########################################################################################################################

DATE_PREDICT = '12.5.2019'
x_predict = np.array([date_to_x(datetime.strptime(DATE_PREDICT,'%d.%m.%Y').date())])

dh=draw_history()
print(dh)

x_train = np.array([draw.x_train  for draw in dh.draws])
y_train = np.array([draw.y_train  for draw in dh.draws])

y_predict = learn_and_predict_sportka2(x_train, y_train, x_predict)

print(y_predict)
print('best numbers for {}\n: {}\n\n'.format(DATE_PREDICT, best_numbers(y_predict, 6)))


print('all numbers\n: {}\n\n'.format( best_numbers(y_predict, 49)))
