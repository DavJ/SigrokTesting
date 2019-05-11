import csv
from datetime import date, datetime
import tensorflow as tf
import numpy as np

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
            self.date = datetime.strptime(row[0],'%d. %m. %Y').date()
            self.week = int(row[2])
            self.week_day = int(row[3])
            self.first = [int(x) for x in row[4:11]]
            self.second = [int(x) for x in row[11:18]]
            print('>OK')
        except:
            print('>ERROR')

    @property
    def x_train(self):
        return date_to_x(self.date)

    @property
    def y_train(self):
        probability_first = np.array([1.0 if  number in self.first else 0 for number in range(1, 49)])
        probability_second = np.array([1.0 if  number in self.first else 0 for number in range(1, 49)])
        return 0.5*(probability_first + probability_second)

def date_to_x(date):
    return np.array([
        date.day / 31.0,
        date.month / 12.0,
        date.year / 2019.0,
        date.weekday() / 6.0
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
        tf.keras.layers.Flatten(input_shape=(7)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(49, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    #model.evaluate(x_test, y_test)

    return model.predict(x_predict)

########################################################################################################################
############################## main program ############################################################################
########################################################################################################################

DATE_PREDICT = '12.5.2019'
x_predict = date_to_x(datetime.strptime(DATE_PREDICT,'%d.%m.%Y').date())

dh=draw_history()
print(dh)

x_train = [draw.x_train  for draw in dh.draws]
y_train = [draw.y_train  for draw in dh.draws]

y_predict = learn_and_predict_sportka(x_train, y_train, x_predict)


print(y_predict)
pass