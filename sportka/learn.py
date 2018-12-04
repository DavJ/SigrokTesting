import csv
from datetime import date, datetime

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
            self.date = datetime.strptime(row[0],'%d. %m. %Y')
            self.week = int(row[2])
            self.week_day = int(row[3])
            self.first = [int(x) for x in row[4:11]]
            self.second = [int(x) for x in row[12:17]]
            print('>OK')
        except:
            print('>ERROR')
    
    
dh=draw_history()
print(dh)
