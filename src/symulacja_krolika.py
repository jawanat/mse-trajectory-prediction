import matplotlib.pyplot as pl  
import numpy as np 
from src.generacja_poczatkowych_polozen import lokalizacja_krolika

krolik = lokalizacja_krolika(poczatkowy_x=0.0, poczatkowy_y=0.0, speed=3.0)

x_wspolrzedne = np.zeros(6)
y_wspolrzedne = np.zeros(6)

for i in range(1,6):
    pozycja = krolik.kolejne_polozenia()
    x_wspolrzedne[i] = pozycja[0]
    y_wspolrzedne[i] = pozycja[1]

pl.plot(x_wspolrzedne, y_wspolrzedne, marker='o', linestyle='-', color='g')
pl.xlabel('Pozycja X królika')
pl.ylabel('Pozycja Y królika')
pl.title('Ścieżka ucieczki królika w 2D')
pl.grid(True)
pl.show()