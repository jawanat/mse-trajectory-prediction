import numpy as np
#położenie królika
rabbit_data = np.array([
    (0.0, 0.00, 0.00),
    (0.5, 0.45, 0.20),
    (1.0, 0.90, 0.05),
    (1.5, 1.50, 0.60),
    (2.0, 2.80, 0.10),
    (2.5, 4.50, 1.20),
    (3.0, 6.80, 0.50),
    (3.5, 9.50, 1.80),
    (4.0, 12.00, 0.20)
])
t=rabbit_data[:,0]
x=rabbit_data[:,1]
y=rabbit_data[:,2]

#nie wiemy jaki stopień wielomianu będzie najleszpy
#dla przykładowego królika
def najlepszy_stopien (t,x,y,max_stopien=500):
    poczatkowy_blad=float('inf')
    najlepszy_stopien=1
    for stopien in range(1,max_stopien+1):
        #wyznaczam wspolrzedne do dwóch równań
        # 1) opisuje położenie względem osi x po t czasie
        # 2) opisuje położenie względem osi y po t czasie
        wspolrzedna_x=np.polyfit(t,x,stopien)
        wspolrzedna_y=np.polyfit(t,y,stopien)

        #tworze model odpowiednio dla x i y
        model_x=np.poly1d(wspolrzedna_x)
        model_y=np.poly1d(wspolrzedna_y)
        #obliczam jakie są wartości zgodnie z przyjętym modelem
        wartosc_x_wedlug_modelu=model_x(t)
        wartosc_y_wedlug_modelu=model_y(t)
        
        #błędy średniokwadratowe
        MSE_x=np.mean((x-wartosc_x_wedlug_modelu)**2)
        MSE_y=np.mean((y-wartosc_y_wedlug_modelu)**2)

        MSE=MSE_x+MSE_y

        if poczatkowy_blad>MSE:
            poczatkowy_blad=MSE
            najlepszy_stopien=stopien
        
    return najlepszy_stopien

print(f"Najlepszy stopień dla Twojego królika to: {najlepszy_stopien(t, x, y)}")