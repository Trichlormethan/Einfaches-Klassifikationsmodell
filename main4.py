# Binärer Sentiment-Klassifikator
# Logistische Regression

# Wir trainieren ein einfaches Modell, das Texte als positiv oder negativ klassifiziert.

import numpy as np  # Paket gebraucht für Vektoren-, Matrizen- und Skalarmultiplikationen durchführen zu können
import pandas as pd  # Dieses Paket benötigen wir für die Extraktion von Daten aus Datentabellen wie Excel
import matplotlib.pyplot as plt # Fehlerkurve
from sklearn.metrics import ConfusionMatrixDisplay # Konfusionmatrix

np.random.seed(12)

# relativer pfad
dateipfad = "sentiment_dataset.xlsx"
# absoluter pfad
# dateipfad = C:\Users\Nutzername\Documents\sentiment_dataset.xlsx

daten = pd.read_excel(dateipfad)

texte = daten["text"].astype(str).tolist()  # alle texte in einer gemeinsamen liste appended in string
labels = daten["label"].values.astype(float)  # alle label Zahlen in ein numpy array appended in float

# print(texte)
# print(labels)


print("Anzahl positive Texte:", int(labels.sum())) # rechnen alle einsen zusammen
print("Anzahl negative Texte:", int(len(labels) - labels.sum()))


# ein einzelner text wird kleingeschrieben und in wörter zerlegt
def text_tokenisieren(text):
    text = str(text)
    text = text.lower()

    tokens = text.split()

    return tokens

# um zu verhindern, dass wir zu lange unnötige dimensionen erhalten, verkürzen wir das vokabular sinnvoll
# wörter die unter 2 mal vorkommen, werden nichts ins vokabular eingetragen
def vokabular_erstellen(alle_texte):
    mindest_anzahl = 2
    dokumentzahl = {}

    for text in alle_texte:
        tokens = text_tokenisieren(text)
        einzigartige_wörter = set(tokens)

        for wort in einzigartige_wörter:
            if wort not in dokumentzahl:
                dokumentzahl[wort] = 0

            dokumentzahl[wort] += 1

    vokabular = {}

    for wort in sorted(dokumentzahl): # vokabular reihenfolge festlegen, wir fixieren es alphabetisch
        if dokumentzahl[wort] >= mindest_anzahl:
            index = len(vokabular)
            vokabular[wort] = index

    dimensionen = len(vokabular)
    # bei mindest_anzahl = 2 enthält unser vokabular 269 wörter (269 dimensionen)
    return vokabular, dimensionen

def sigmoid(z):
    score = 1 / (1 + np.exp(-z))

    return score

def vorhersagen_berechnen(x, w, bias):

    z = np.dot(x, w) + bias # führt die skalarmultiplikation durch

    y_hat = sigmoid(z)

    return y_hat


def log_loss(richtige_labels, vorhersagen):
    # epsiolin ist eine kleine zahl, die verhindert, dass ln(0) berechnet wird (=undefiniert)
    epsilon = 0.000000001
    # L = -(y * ln(y_hat) + (1-y) * ln(1-y_hat))

    positiver_teil = (richtige_labels * np.log(vorhersagen + epsilon))
    negativer_teil = ((1 - richtige_labels) * np.log(1 - vorhersagen + epsilon))

    kosten = -np.mean(positiver_teil + negativer_teil) # minus zeichen kommt zusätzlich in

    return kosten


def text_zu_binärvektor(text, vokabular):
    vektor = np.zeros(len(vokabular), dtype=float)

    tokens = text_tokenisieren(text)
    for wort in tokens:
        if wort in vokabular: # zum prüfen ob das wort im vokabular ist
            index = vokabular[wort]
            vektor[index] = 1

    return vektor

def dokument_begriffsmatrix(alle_texte, vokabular):
    # wir machen eine begriffsmatrix aus allen erhaltenen vektoren (nur zur veranschaulichung)
    alle_vektoren = []

    for text in alle_texte:
        vektor = text_zu_binärvektor(text, vokabular)
        alle_vektoren.append(vektor)

    matrix = np.array(alle_vektoren) # jetzt haben wir alle vektoren nacheinander.

    spalten = [None] * len(vokabular) # spalten als liste muss anscheinend so geschrieben werden

    for wort in vokabular:
        index = vokabular[wort]
        spalten[index] = wort


    tabelle = pd.DataFrame(matrix, columns=spalten)
    pd.set_option("display.max_rows", None)

    # return matrix
    # wir transponieren hier die matrix.
    return tabelle.T # .to_string() #gefährlich

def gradient_weight(x, richtige_labels, vorhersage):
    vorhersageabweichung = vorhersage - richtige_labels # y hut - y
    gradient_w = vorhersageabweichung * x
    return gradient_w

def gradient_bias(richtige_labels, vorhersage):
    vorhersageabweichung = vorhersage - richtige_labels
    return vorhersageabweichung


def train_test_split(texte, labels): # diese funktion ist für den 80/20 split
    test_anteil = 100
    anzahl = len(texte) # 500

    indexe = np.arange(anzahl)
    np.random.shuffle(indexe)     # zufällig mischen, damit die aufteilung fair ist

    grenze = int(anzahl - test_anteil)   # grenze legen wir bei 400 fest

    train_indexe = indexe[0:grenze]
    test_indexe = indexe[grenze:]

    train_texte = []
    for i in train_indexe:
        train_texte.append(texte[i])

    test_texte = []
    for i in test_indexe:
        test_texte.append(texte[i])

    train_labels = labels[train_indexe]   # labels ist ein numpy array, das geht direkt
    test_labels = labels[test_indexe]

    return train_texte, test_texte, train_labels, test_labels

def training(x_traininings_texte, y_trainings_label):
    alpha = 0.25
    epochen = 250

    # die trainingstexte sind eine liste von binärvektoren und trainingslabel ein array mit den passenden labels
    anzahl_wörter = len(x_traininings_texte[0])
    w = np.zeros(anzahl_wörter)
    bias = 0.0
    x_matrix = np.array(x_traininings_texte)
    verlauf_loss = [] # damit wir den fehler speichern um für später die kurve illustrieren zu können

    for epoche in range(epochen):
        reihenfolge = np.random.permutation(len(x_traininings_texte))
        # permutation() shuffled and arranged die werte in einem array statt shuffle() und danach arange()

        for i in reihenfolge:
            x_i = x_traininings_texte[i]
            y_i = y_trainings_label[i]

            y_hut_i = vorhersagen_berechnen(x_i, w, bias)

            grad_w = gradient_weight(x_i, y_i, y_hut_i)
            grad_b = gradient_bias(y_i, y_hut_i)

            w = w - (alpha * grad_w)
            bias = bias - (alpha * grad_b)

        # loss berechnungen durchführen, dafür brauchen wir die trainingstexte als matrix
        y_hat_alle = vorhersagen_berechnen(x_matrix, w, bias)  # ergebnis: vektor mit 400 werten
        aktueller_loss = log_loss(y_trainings_label, y_hat_alle)

        verlauf_loss.append(aktueller_loss)

        if epoche % 10 == 0:
            print(f"Epoche {epoche}: Fehler = {aktueller_loss:.4f}")

    return w, bias, verlauf_loss

def genauigkeit(vorhersagen, richtige_labels):

    richtige_antworten = 0
    anzahl_texte = len(richtige_labels)

    for i in range(anzahl_texte):
        if vorhersagen[i] >= 0.5:
            vorhergesagte_klasse = 1
        else:
            vorhergesagte_klasse = 0

        if vorhergesagte_klasse == richtige_labels[i]:
            richtige_antworten = richtige_antworten + 1

    return richtige_antworten / anzahl_texte

def konfusionsmatrix(vorhersagen, richtige_labels):
    richtig_positiv = 0
    falsch_negativ = 0
    falsch_positiv = 0
    richtig_negativ = 0

    anzahl_texte = len(richtige_labels)

    for i in range(anzahl_texte):
        if vorhersagen[i] >= 0.5:
            vorhergesagte_klasse = 1
        else:
            vorhergesagte_klasse = 0

        if vorhergesagte_klasse == richtige_labels[i]:
            if vorhergesagte_klasse == 1:
                richtig_positiv = richtig_positiv + 1
            else:
                richtig_negativ = richtig_negativ + 1
        else:
            if vorhergesagte_klasse == 1:
                falsch_positiv = falsch_positiv + 1
            else:
                falsch_negativ = falsch_negativ + 1

    return richtig_positiv, falsch_negativ, falsch_positiv, richtig_negativ


def konfusionsmatrix_zeichnen(richtig_positiv, falsch_negativ, falsch_positiv, richtig_negativ):
    matrix = np.array([
        [richtig_negativ, falsch_positiv],
        [falsch_negativ, richtig_positiv]
    ])

    cm = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["negativ", "positiv"])
    cm.plot(cmap="Blues")
    plt.title("Konfusionsmatrix")
    plt.show()


def fehlerkurve(verlauf_loss):
    plt.plot(verlauf_loss)
    plt.xlabel("Epoche")
    plt.ylabel("Log-Loss")
    plt.title("Fehlerverlauf beim Training")
    plt.show()

def text_klassifizieren(neuer_text, vokabular, gelernte_gewichte, gelernter_bias):
    x = text_zu_binärvektor(neuer_text, vokabular)

    wahrscheinlichkeit = vorhersagen_berechnen(x, gelernte_gewichte, gelernter_bias)

    if wahrscheinlichkeit >= 0.5:
        klasse = "positiv"
    else:
        klasse = "negativ"

    return klasse, wahrscheinlichkeit

#---------------------Ausführung---------------------

# 1. trainings- und testdaten aufteilen

train_texte, test_texte, train_labels, test_labels = train_test_split(texte, labels)

"""
train_texte, _, _, _ = train_test_split(texte, labels)
_, test_texte, _, _ = train_test_split(texte, labels)
_, _, train_labels, _ = train_test_split(texte, labels)
_, _, _, test_labels = train_test_split(texte, labels)
"""

# 2. vokabular nur aus den trainingstexten bauen
vokabular, dimensionen = vokabular_erstellen(train_texte)

print("Anzahl Dimensionen (Vokabulargrösse) ", dimensionen)

# 3. beide gruppen mit demselben vokabular in binärvektoren umwandeln
# zuerst waren sie als strings in eine liste gesammelt, jetzt werden sie in binärvektoren vektorisiert
x_train = []
for text in train_texte:
    x_train.append(text_zu_binärvektor(text, vokabular))

x_test = []
for text in test_texte:
    x_test.append(text_zu_binärvektor(text, vokabular))

# 3.5 dokument_begriffsmatrix visualisieren
matrix = dokument_begriffsmatrix(train_texte, vokabular)
print(matrix)

# 4. modell trainieren
gelernte_gewichte, gelernter_bias, verlauf_loss = training(x_train, train_labels)


# 5. da das training fertig ist und die gelernten parameter feststehen, wird das
# modell auf den testdaten ausprobiert und gewertet
x_test_matrix = np.array(x_test)

vorhersagen_test = vorhersagen_berechnen(x_test_matrix, gelernte_gewichte, gelernter_bias)



finale_genauigkeit = genauigkeit(vorhersagen_test, test_labels)
print(f"Testgenauigkeit: {finale_genauigkeit:.2%}")

richtig_positiv, falsch_negativ, falsch_positiv, richtig_negativ = konfusionsmatrix(vorhersagen_test, test_labels)

# 6. widgets anzeigen
fehlerkurve(verlauf_loss)
konfusionsmatrix_zeichnen(richtig_positiv, falsch_negativ, falsch_positiv, richtig_negativ)

#-------------------------------------------------------------------------------------------------------------

programm_läuft = True

while programm_läuft:
    neuer_text = input("Text eingeben: ")

    if neuer_text.lower() == "q":
        programm_läuft = False
    else:

        klasse, wahrscheinlichkeit = text_klassifizieren(neuer_text, vokabular, gelernte_gewichte, gelernter_bias)
        print(f"Klasse: {klasse} (Wahrscheinlichkeit: {wahrscheinlichkeit:.2f})")



#----------------------- testing -----------------------

# print(dokument_begriffsmatrix(texte, vokabular_erstellen(texte, mindest_anzahl=2)))
# test_daten, _, _, _ = train_test_split(texte, labels, test_anteil=100) # retourniere nur einen wert
# print(test_daten)
