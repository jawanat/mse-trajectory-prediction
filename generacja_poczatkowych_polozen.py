import numpy as np
from math import pi
#na początku muszę wprowadzić pewne założenia
#czyli z jaką prędkością królik biegnie oraz
#jaki będzie szum odnośnie jego trasy
#v=3.0
#szum=0.1
class lokalizacja_krolika:
    def __init__(self,poczatkowy_x=0.0,poczatkowy_y=0.0,speed=3.0,szum=0.1):
         self.polozenie=np.array([poczatkowy_x,poczatkowy_y])
         self.poczatkowy_kat=np.random.uniform(0,2*pi)
         self.predkosc=speed
         self.szum=szum
         self.wektor_przemieszczenia=np.array([np.sin(self.poczatkowy_kat),np.cos(self.poczatkowy_kat)])*self.predkosc

    def kolejne_polozenia(self):
        kat_obrotu=np.random.normal(0,self.szum)
        macierz_przemieszczenia = np.array([
            [np.cos(kat_obrotu), -1 * np.sin(kat_obrotu)],
            [np.sin(kat_obrotu), np.cos(kat_obrotu)]
        ])
        self.wektor_przemieszczenia = macierz_przemieszczenia.dot(self.wektor_przemieszczenia)
        self.polozenie=self.polozenie+self.wektor_przemieszczenia

        return(self.polozenie)
    
