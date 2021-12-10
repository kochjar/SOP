# Der importeres nødvendige biblioteker
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Æstetiske parametre for grafernes udseende
plt.style.use('seaborn-poster')
matplotlib.rc('lines', linewidth=1.5, color='r')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Segoe UI'

# Funktion af den diskrete fouriertransformation
def DFT(x):

    # Funktionen tager et array 'x' og returnere dets fourierværdier
    
    N = len(x) # Længden af array'et
    n = np.arange(N) # Der laves et array fra 0 til N
    k = n.reshape((N, 1)) # Længden af arrayet N indsættes i et 1-dimensionelt array: k = [N]
    e = np.exp(-2j * np.pi * k * n / N) # e sættes til at være det eksponentielle led i DFT ligningen

    X = np.dot(e, x) # e ganges med alle værdierne i arrayet x

    return X

# Funktion af den inverse diskrete fouriertransformation
def IDFT(x):

    # Funktionen tager et array 'x' af DFT-værdier fra frekvensdomænet og 
    # returnere dets værdier i tidsdomænet

    N = len(x) # Længden af array'et
    n = np.arange(N) # Der laves et array fra 0 til N
    k = n.reshape((N, 1)) # Længden af arrayet reshapes til et vertikal 1-dimensionelt array

    e = np.exp(2j * np.pi * k * n / N) # e sættes til at være det eksponentielle led i IDFT ligningen

    # e ganges med alle værdierne i arrayet x samt ganges med den reciprokke værdi af antallet af prøver
    x_n = 1 / N * np.dot(e, x) 

    return x_n

# LowPass-filteret filtrerer alle frekvenser 'x', der er højere end 'n' Hz
def LowPassFilter(x, n):
    # En for-løkke sætter alle frekvenser højere end n lig med 0
    for i in range(n, len(x)):
        x[i] = 0
    return x

# HighPass-filteret filtrerer alle frekvenser 'x', der er laverel end 'n' Hz
def HighPassFilter(x, n):
    # En for-løkke sætter alle frekvenser lavere end n lig med 0
    for i in range(0, n):
        x[i] = 0
    return x

# AmpFilter-funktionen filtrerer  alle frekvenser 'x', der ikke overstiger amplitudeværdier over 'n'
def AmpFilter(x, n):
    # Der laves et array 'indices' der består af true og false 
    # alt efter om abs(x) er over amplitudegrænsen
    indices = abs(x) > n
    # inputtet 'x' ganges med 'indices' - derved forsvicnder de værdier
    # som overstiger grænsen 'n'
    x = x * indices
    return x


sr = 500 # Antal prøver
ts = 1.0/sr # Prøveudtagningsinterval

# Der laves en liste over x værdier mellem 0 og 1 med et interval på ts
t = np.arange(0, 1, ts)

# Der laves en funktion x(t) bestående af 3 sinusbølger af forskellige amplituder og frekvenser
freq = 1
x = 3*np.sin(2*np.pi*freq*t)
freq = 4
x += 1*np.sin(2*np.pi*freq*t)
freq = 14
x += 1.5*np.sin(2*np.pi*freq*t)

x_clean = x # Dette signal bliver det rene signal

x = x + 2*np.random.randn(len(t)) # Der tilføjes støj til signalet


# Der laves et plot over henholdsvis det rene signal og det urene signal
plt.figure(figsize=(8, 6)) # Plottets størrelse defineres
plt.title("Graf over rent og urent signal") # Plottets titelnavn defineres
plt.plot(t, x, color='#000000', linewidth=0.5, label="Signal med støj") # Der plottes det urene signal over tid med sort
plt.plot(t, x_clean, color='#bd0f1e', linewidth=2, label="Rent signal") # Der plottes det rene signal over tid med rødt
plt.legend() # Grafforklaring vises
plt.xlabel('t') # Aksetitel for x-akse
plt.ylabel('x(t)') # Aksetitel for y-akse
plt.savefig('Figur_1.png', dpi=200) # Billede af graf gemmes
plt.show() # Plottet vises


X = DFT(x) # DFT af x beregnes

# Frekvenserne beregnes
N = len(X)
freq = np.arange(N)

# Der laves et plot af frekvensdomænet for funktionen x(t)
plt.figure(figsize=(8, 6)) # Plottets størrelse defineres
plt.stem(freq, abs(X), 'b', markerfmt=" ", basefmt="-b") # x- og y-værdier defineres
plt.xlabel('Frekvens [ Hz ]') # Aksetitel for x-akse
plt.ylabel('DFT Amplitude |X(freq)|') # Aksetitel for y-akse
plt.savefig('Figur_2.png', dpi=200) # Billede af graf gemmes
plt.show() # Plottet vises

# Nyquist-frekvensen bliver fundet ved at tage N modulus 2 - altså halvdelen af den oprindelige antal prøver
n_oneside = N//2 
# Der bliver fundet frekvenser fra denne halve side
f_oneside = freq[:n_oneside]

# DFT værdierne skal normaliseres ved at dele outputtet med antallet af prøver
X_oneside = X[:n_oneside]/n_oneside

# Her tilvælges evt. et filter 
#X_filtered = LowPassFilter(X, 15)
#X_filtered = HighPassFilter(X, 50)
X_filtered = AmpFilter(X_oneside, 0.5)

# Der laves plot over det normaliserede frekvens domæne før og efter filtrerng
plt.figure(figsize=(12, 6)) # Plottets størrelse defineres
plt.subplot(121) 
plt.stem(f_oneside, abs(X_oneside), 'b',
         markerfmt=" ", basefmt="-b") # x- og y-værdier defineres for plottet
plt.xlabel('Frekvens [ Hz ]') # Aksetitle for x-akse
plt.ylabel('Normaliseret DFT Amplitude |X(freq)|') # Aksetitle for y-akse
plt.title("Før filtrering") 
plt.subplot(122)
plt.stem(f_oneside, abs(X_filtered), 'b',
         markerfmt=" ", basefmt="-b")  # x- og y-værdier defineres for plottet
plt.xlabel('Frekvens [ Hz ]') # Aksetitle for x-akse
plt.tight_layout()
plt.title("Efter filtrering")
plt.savefig('Figur_3.png', dpi=200) # Billede af plot gemmes
plt.show() # Plottet vises


reverse = IDFT(X_filtered) # Den inverse diskrete fouriertransformation findes

# Den filtrede Funktion x(t) samt det rene signal af x(t) plottes
plt.figure(figsize=(8, 6)) # Plottets størrelse
plt.title("Filtreret signal af x(t)") 
#plt.plot(t, reverse, 'black')
plt.plot(t, reverse, color='#000000', linewidth=1.5, label="Filtreret signal")  # Det filtrede signal over tid markeret med med sort
plt.plot(t, x_clean, color='#bd0f1e', linewidth=1.5, label="Rent signal") # Det rene signal over tid markeret med rød
plt.legend() # Grafforklaring vises 
plt.xlabel('t') # Aksetitel for x-akse
plt.ylabel('x(t)') # Aksetitel for y-akse
plt.savefig('Figur_4.png', dpi=200) # Billede af graf gemmes
plt.show() # Plot vises
