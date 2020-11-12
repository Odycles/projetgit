import datetime
#Lolane est passée par là
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean,median,pvariance,pstdev
from numpy import quantile
data = pd.read_csv(r"C:\Users\odyss\Documents\Cours\Informatique\EIVP_KM.csv", sep = ';', header = None)

#anomalies : bruit max sur le capteur 2, capteur 5 qui déconne pendant 2 jours

ID = data[1]
ID.pop(0)

noise = data[2]
noise.pop(0)

temp = data[3]
temp.pop(0)

humidity = data[4]
humidity.pop(0)

lum = data[5]
lum.pop(0)

co2 = data[6]
co2.pop(0)

sent_at = data[7]
sent_at.pop(0)

from statistics import mean,median,pvariance,pstdev
from numpy import quantile

def moyenne(num_capteur, donnee):

    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4

    liste_capteur = tableau[num_capteur-1]
    data = []

    for terme in liste_capteur :
        data.append(float(terme[num_donnee]))

    moyenne = mean(data)
    return moyenne


def mediane(num_capteur, donnee):

    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4

    liste_capteur = tableau[num_capteur-1]
    data = []

    for terme in liste_capteur :
        data.append(float(terme[num_donnee]))

    mediane = median(data)
    return mediane

def quartile(num_capteur, donnee):

    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4

    liste_capteur = tableau[num_capteur-1]
    data = []

    for terme in liste_capteur :
        data.append(float(terme[num_donnee]))

    Q1 = quantile(data,0)
    Q3 = quantile(data,1)

    return Q1,Q3

def variance(num_capteur, donnee):

    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4

    liste_capteur = tableau[num_capteur-1]
    data = []

    for terme in liste_capteur :
        data.append(float(terme[num_donnee]))

    variance = pvariance(data)
    return variance


def ecart_type(num_capteur, donnee):

    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4

    liste_capteur = tableau[num_capteur-1]
    data = []

    for terme in liste_capteur :
        data.append(float(terme[num_donnee]))

    ecart_type= pstdev(data)
    return ecart_type


def max_donnee(num_capteur, donnee):


    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4

    liste_capteur = tableau[num_capteur-1]
    data = []

    for terme in liste_capteur :
        data.append(float(terme[num_donnee]))

    maxi = max(data)
    return maxi

def min_donnee(num_capteur, donnee):

    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4

    liste_capteur = tableau[num_capteur-1]
    data = []

    for terme in liste_capteur :
        data.append(float(terme[num_donnee]))

    mini = min(data)
    return mini

def correlation(num_capteur,donneeA,donneeB):
    ecartypeA = ecart_type(num_capteur,donneeA)
    ecartypeB = ecart_type(num_capteur,donneeB)

    liste_capteur = tableau[num_capteur-1]
    dataA=[]
    dataB=[]

    if donneeA == "noise" :
        num_donneeA = 0
    elif donneeA == "temp":
        num_donneeA = 1
    elif donneeA == "humidity":
        num_donneeA = 2
    elif donneeA == "lum" :
        num_donneeA = 3
    elif donneeA == "co2":
        num_donneeA = 4


    if donneeB == "noise" :
        num_donneeB = 0
    elif donneeB == "temp":
        num_donneeB = 1
    elif donneeB == "humidity":
        num_donneeB = 2
    elif donneeB == "lum" :
        num_donneeB = 3
    elif donneeB == "co2":
        num_donneeB = 4

    for terme in liste_capteur :
            dataA.append(float(terme[num_donneeA]))
            dataB.append(float(terme[num_donneeB]))



    covariance = np.cov(dataA,dataB)[0][1]
    correl = covariance/(ecartypeA*ecartypeB)
    return correl

def all_moyenne(num):
    moyenne_temp = moyenne(num, "temp")
    moyenne_noise = moyenne(num, "noise")
    moyenne_co2 = moyenne(num, "co2")
    moyenne_humidity = moyenne(num, "humidity")
    moyenne_lum = moyenne(num, "lum")
    print( 'Moyennes Capt ',num,' :\n   Température = ',moyenne_temp,"\n   Noise = ",moyenne_noise,"\n   co2 = ",moyenne_co2,"\n   humidity = ", moyenne_humidity,"\n   lum = ",moyenne_lum,'\n')

def all_max(num):
    max_temp = max_donnee(num, "temp")
    max_noise = max_donnee(num, "noise")
    max_co2 = max_donnee(num, "co2")
    max_humidity = max_donnee(num, "humidity")
    max_lum = max_donnee(num, "lum")
    print( 'Maxs Capt ',num,' :\n   Température = ',max_temp,"\n   Noise = ",max_noise,"\n   co2 = ",max_co2,"\n   humidity = ", max_humidity,"\n   lum = ",max_lum,'\n')

def all_min(num):
    min_temp = min_donnee(num, "temp")
    min_noise = min_donnee(num, "noise")
    min_co2 = min_donnee(num, "co2")
    min_humidity = min_donnee(num, "humidity")
    min_lum = min_donnee(num, "lum")
    print( 'Mins Capt ',num,' :\n   Température = ',min_temp,"\n   Noise = ",min_noise,"\n   co2 = ",min_co2,"\n   humidity = ", min_humidity,"\n   lum = ",min_lum,'\n')


def all_mediane(num):
    mediane_temp = mediane(num, "temp")
    mediane_noise = mediane(num, "noise")
    mediane_co2 = mediane(num, "co2")
    mediane_humidity = mediane(num, "humidity")
    mediane_lum = mediane(num, "lum")
    print( 'Medianes Capt ',num,' :\n   Température = ',mediane_temp,"\n   Noise = ",mediane_noise,"\n   co2 = ",mediane_co2,"\n   humidity = ", mediane_humidity,"\n   lum = ",mediane_lum,'\n')


def all_variance(num):
    variance_temp = variance(num, "temp")
    variance_noise = variance(num, "noise")
    variance_co2 = variance(num, "co2")
    variance_humidity = variance(num, "humidity")
    variance_lum = variance(num, "lum")
    print( 'Variances Capt ',num,' :\n   Température = ',variance_temp,"\n   Noise = ",variance_noise,"\n   co2 = ",variance_co2,"\n   humidity = ", variance_humidity,"\n   lum = ",variance_lum,'\n')


def all_ecart_type(num):
    ecart_type_temp = ecart_type(num, "temp")
    ecart_type_noise = ecart_type(num, "noise")
    ecart_type_co2 = ecart_type(num, "co2")
    ecart_type_humidity = ecart_type(num, "humidity")
    ecart_type_lum = ecart_type(num, "lum")
    print( 'Ecart_types Capt ',num,' :\n   Température = ',ecart_type_temp,"\n   Noise = ",ecart_type_noise,"\n   co2 = ",ecart_type_co2,"\n   humidity = ", ecart_type_humidity,"\n   lum = ",ecart_type_lum,'\n \n \n \n')


def all_stats():
    for i in range (1,7):
        all_moyenne(i)
        all_max(i)
        all_min(i)
        all_mediane(i)
        all_variance(i)
        all_ecart_type(i)

from math import log,e
#cst pour alpha
a = 17.27
b = 237.7 #en degré celsius

def alpha(T,phi): #phi = humidity
    phi_p = float(phi) /100
    fraction = (a * T)/(b+T)
    ln = log(phi_p)
    alpha = fraction + ln
    return alpha

def T_rosee(T,phi):
    num = b * alpha(T,phi)
    denom = a - alpha(T,phi)
    return num/denom

def humidex(T,phi):
    expo = 5417.7530*((1/273.16)-(1/(273.15+T_rosee(T,phi))))
    crochet = (6.11*(e**(expo)))-10
    return T + 0.5555*crochet

def separation_capteur(ID, noise, temp, humidity, lum, co2, sent_at):
        tableau = []
        i = 1
        for k in range (1,7):
            L = []
            while int(ID[i]) == k and i < 7880 :
                donnees = []
                donnees.append(noise[i])
                donnees.append(temp[i])
                donnees.append(humidity[i])
                donnees.append(lum[i])
                donnees.append(co2[i])
                donnees.append(sent_at[i])
                L.append(donnees)
                i = i + 1
            tableau.append(L)
        return tableau

tableau = separation_capteur(ID, noise, temp, humidity, lum, co2, sent_at)




def def_time(liste_date) :
    temps = []
    for date in liste_date :
        annee = date[0:4]
        mois = date [5:7]
        if mois[0] == 0 :
            mois.pop(0)
        jour = date[8:10]
        if jour[0] == 0 :
            jour.pop(0)
        heure = date[11:13]
        if heure[0] == 0 :
            heure.pop(0)
        minute = date[14:16]
        if minute[0] == 0 :
            minute.pop(0)
        seconde = date[17:19]
        if seconde[0] == 0 :
            seconde.pop(0)
        annee,mois,jour,heure,minute,seconde = int(annee),int(mois),int(jour),int(heure),int(minute),int(seconde)
        t = datetime.datetime(annee,mois,jour,heure,minute,seconde)
        temps.append(t)
    return temps




def trace(num_capteur,donnee,start_date,end_date):
    humid_ex = 0
    k = 0
    liste_capteur = tableau[num_capteur-1]
    tps = []
    data = []
    L_tps = []

    for capt in tableau :
        for point in capt :
            T = float(point[1])
            phi = float(point[2])
            humid_ex = humidex(T,phi)
            point.append(humid_ex)


    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4
    elif donnee == "humidex":
        num_donnee = 6

    if start_date != None and end_date != None :

        for terme in liste_capteur :
                tps.append(terme[5])
                data.append(float(terme[num_donnee]))

        tps = def_time(tps)


        annee = start_date[0:4]
        mois = start_date [5:7]
        if mois[0] == 0 :
            mois.pop(0)
        jour = start_date[8:10]
        if jour[0] == 0 :
            jour.pop(0)
        annee,mois,jour,heure,minute,seconde = int(annee),int(mois),int(jour),0,0,0
        t1 = datetime.datetime(annee,mois,jour,heure,minute,seconde)

        annee = end_date[0:4]
        mois = end_date [5:7]
        if mois[0] == 0 :
            mois.pop(0)
        jour = end_date[8:10]
        if jour[0] == 0 :
            jour.pop(0)
        annee,mois,jour,heure,minute,seconde = int(annee),int(mois),int(jour),0,0,0
        t2 = datetime.datetime(annee,mois,jour,heure,minute,seconde)


        t=tps[k]
        while t1 > t :
            print(k)
            t = tps[k]
            k+=1
        ite_start_date = k
        while t2 > t :
            t = tps[k]
            k+=1
        ite_end_date = k

        tps_origin = tps[0]

        tps = tps[ite_start_date:ite_end_date]

        for terme in tps :
            delta_origin = tps[0]-tps_origin
            delta = terme - tps[0]
            seconde = delta.total_seconds() + delta_origin.total_seconds()
            jour = (seconde /3600) /24
            L_tps.append(jour)
        data = []
        for terme in liste_capteur[ite_start_date:ite_end_date] :
                data.append(float(terme[num_donnee]))



        maxi = max(data)
        mini = min(data)
        moyenne = round(mean(data),2)
        variance = round(pvariance(data),2)
        ecarttype = round(pstdev(data),2)
        mediane = median(data)

        plt.plot(L_tps,data)
        plt.ylabel(donnee)
        plt.xlabel('jours depuis le premier jour')

        ajoutx = (max(L_tps)-min(L_tps))*0.22

        plt.annotate(maxi,xy = (min(L_tps)+ajoutx,mini))
        plt.annotate('max =',xy = (min(L_tps),mini))

        plt.annotate(mini,xy = (min(L_tps)+ajoutx ,mini+0.5))
        plt.annotate('min =',xy = (min(L_tps),mini+0.5))

        plt.annotate(moyenne,xy = (min(L_tps)+ajoutx ,mini+1))
        plt.annotate('moyenne =',xy = (min(L_tps),mini+1))

        plt.annotate(variance,xy = (min(L_tps)+ajoutx ,mini+1.5))
        plt.annotate('variance =',xy = (min(L_tps),mini+1.5))

        plt.annotate(ecarttype,xy = (min(L_tps)+ajoutx ,mini+2))
        plt.annotate('ecart type =',xy = (min(L_tps),mini+2))

        plt.annotate(mediane,xy = (min(L_tps)+ajoutx ,mini+2.5))
        plt.annotate('mediane =',xy = (min(L_tps),mini+2.5))


        plt.show()









    else :
        for terme in liste_capteur :
                tps.append(terme[5])
                data.append(float(terme[num_donnee]))
        tps = def_time(tps)
        L_tps = []

        for terme in tps :
            delta = terme - tps[0]
            seconde = delta.total_seconds()
            jour = (seconde /3600) /24
            L_tps.append(jour)

        maxi = max(data)
        mini = min(data)
        moyenne = round(mean(data),2)
        variance = round(pvariance(data),2)
        ecarttype = round(pstdev(data),2)
        mediane = median(data)

        plt.plot(L_tps,data)
        plt.ylabel(donnee)
        plt.xlabel('jours depuis le premier jour')

        ajoutx = (max(L_tps)-min(L_tps))*0.22

        plt.annotate(maxi,xy = (min(L_tps)+ajoutx,mini))
        plt.annotate('max =',xy = (min(L_tps),mini))

        plt.annotate(mini,xy = (min(L_tps)+ajoutx ,mini+0.5))
        plt.annotate('min =',xy = (min(L_tps),mini+0.5))

        plt.annotate(moyenne,xy = (min(L_tps)+ajoutx ,mini+1))
        plt.annotate('moyenne =',xy = (min(L_tps),mini+1))

        plt.annotate(variance,xy = (min(L_tps)+ajoutx ,mini+1.5))
        plt.annotate('variance =',xy = (min(L_tps),mini+1.5))

        plt.annotate(ecarttype,xy = (min(L_tps)+ajoutx ,mini+2))
        plt.annotate('ecart type =',xy = (min(L_tps),mini+2))

        plt.annotate(mediane,xy = (min(L_tps)+ajoutx ,mini+2.5))
        plt.annotate('mediane =',xy = (min(L_tps),mini+2.5))
        plt.show()



def detect_anomalie(num_capteur,donnee):
    k = 0
    liste_capteur = tableau[num_capteur-1]
    tps = []
    data = []
    L_tps = []


    if donnee == "noise" :
        num_donnee = 0
    elif donnee == "temp":
        num_donnee = 1
    elif donnee == "humidity":
        num_donnee = 2
    elif donnee == "lum" :
        num_donnee = 3
    elif donnee == "co2":
        num_donnee = 4



    for terme in liste_capteur :
            tps.append(terme[5])
            data.append(float(terme[num_donnee]))
    tps = def_time(tps)
    L_tps = []

    for terme in tps :
        delta = terme - tps[0]
        seconde = delta.total_seconds()
        jour = (seconde /3600) /24
        L_tps.append(jour)

    liste_ano=[]
    k=0
    for t in L_tps :
        ecart = L_tps[k]-L_tps[k-1]

        if ecart > 0.1 :
            liste_ano.append([L_tps[k-1],L_tps[k],data[k],data[k-1]])

        k+=1


    plt.plot(L_tps,data)

    if liste_ano != [] :

        for ano in liste_ano :
            plt.plot([ano[0],ano[1]],[ano[3],ano[2]],'r-',label="anomalie")
            plt.legend()
    else :
        print ("Il n'y a pas d'anomalie ")

    plt.ylabel(donnee)
    plt.show()



##

print ("Bonjour")
print ("Vous pouvez soit : 1. tracer une courbe   2. tracer une courbe entre deux dates  3.detecter les anomalies")
print ("donner le numéro de votre choix")
choix = input()

if choix == "1" :
    print("choisissez une donnée à tracer : temp, humidity, co2, noise, lum, humidex")
    choix_donnee = input()
    print("choisissez un capteur : 1, 2, 3, 4, 5")
    choix_num = input()
    trace(5,choix_donnee,None,None)

if choix == "2" :
    print("choisissez une donnée à tracer : temp, humidity, co2, noise, lum, humidex")
    choix_donnee = input()
    print("choisissez un capteur : 1, 2, 3, 4, 5")
    choix_num = int(input())
    print("choisissez une date de début sous la forme YYYY-MM-DD entre le 2019-08-11 et le 2019-08-25")
    date_debut = str(input())
    print("choisissez une date de fin sous la forme YYYY-MM-DD entre le 2019-08-11 et le 2019-08-25")
    date_fin = str(input())
    trace(choix_num,choix_donnee,date_debut,date_fin)

if choix == "3" :
    print("les anomalies détectées vont être tracées en rouge")
    print("choisissez un capteur a diagnostiquer : 1, 2, 3, 4, 5")
    choix_num = int(input())
    print("choisissez une donnée à tracer : temp, humidity, co2, noise, lum, humidex")
    choix_donnee = input()
    detect_anomalie(choix_num,choix_donnee)









































