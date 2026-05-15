***

# MSE Trajectory Prediction: Wolf & Rabbit Simulation

***

Projekt badawczy realizowany w ramach przedmiotu Metody Optymalizacji. Celem jest analiza zastosowania błędu średniokwadratowego (**Mean Squared Error**) w procesie uczenia sieci neuronowej do przewidywania trajektorii obiektów w środowisku stochastycznym.

***

### **Autorzy**
- ***Marta Marcinkowska***
- ***Jakub Wanat***

***

## Opis projektu
Symulacja przedstawia interakcję dwóch aktorów: **Królika** poruszającego się ruchem losowym (skorelowany random walk) oraz **Wilka**, który wykorzystuje sieć neuronową (Multi-layer Perceptron) do estymacji przyszłego położenia ofiary.

Kluczowym elementem projektu jest optymalizacja wag sieci poprzez minimalizację funkcji kosztu MSE:
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

***

## Technologie
* **Język:** Python 3.13
* **Biblioteki ML:** Scikit-learn (MLPRegressor), NumPy
* **Wizualizacja:** Matplotlib 
* **Dokumentacja:** LaTeX Beamer

***

## Instalacja i uruchomienie
Dzięki zastosowaniu `uv`, konfiguracja środowiska jest szybka i powtarzalna:

1. Zainstaluj `uv`:
```bash
   pip install uv
```
2. Sklonuj repozytorium:
```bash
    git clone https://github.com/jawanat/mse-trajectory-prediction.git
```
3. Pobierz wirtualne środowisko wraz z wykorzystywanymi bibliotekami:
```bash
    uv sync
```

***