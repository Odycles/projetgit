import datetime
import calendar
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean,median,pvariance,pstdev
from numpy import quantile
data = pd.read_csv(r"C:\Users\odyss\Documents\Cours\Informatique\EIVP_KM.csv", sep = ';', header = None)


def moyenne_liste(L):
    somme = 0
    for terme in L :
        somme+=terme
    return somme/len(L)

def mediane_liste(L):
    return sorted(L)[len(L)//2]

def variance_liste(L):
    m=moyenne_liste(L)
    v=0
    for k in range(0,len(L)):
        v=v+(L[k]-m)**2
    return(v/len(L))

def cov(L1, L2):

    if len(L1) != len(L2):
        return

    moyenne_L1 = moyenne_liste(L1)
    moyenne_L2 = moyenne_liste(L2)
    somme = 0

    for i in range(0, len(L1)):
        somme += ((L1[i] - moyenne_L1) * (L2[i] - moyenne_L2))

    return somme/(len(L1)-1)



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

    moyenne = moyenne_liste(data)
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

    mediane = mediane_liste(data)
    return mediane


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

    variance = variance_liste(data)
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

    ecart_type= variance_liste(data)**0.5
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



    covariance = cov(dataA,dataB)
    correl = covariance/(ecartypeA*ecartypeB)
    return round(correl,2)

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

    for terme in liste_capteur :
            tps.append(terme[5])
            terme.append(humidex(float(terme[1]),float(terme[2])))
            data.append(float(terme[num_donnee]))
    tps = def_time(tps)
    L_tps = []


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
    plt.xlabel('jours depuis le premier jour de mesure')

    ajoutx = (max(L_tps)-min(L_tps)) * 0.22
    ajouty = (max(data) - min(data)) * 0.09

    plt.annotate(maxi,xy = (min(L_tps)+ajoutx,mini))
    plt.annotate('max =',xy = (min(L_tps),mini))

    plt.annotate(mini,xy = (min(L_tps)+ajoutx ,mini + ajouty))
    plt.annotate('min =',xy = (min(L_tps),mini + ajouty))

    plt.annotate(moyenne,xy = (min(L_tps)+ajoutx ,mini+(ajouty*2)))
    plt.annotate('moyenne =',xy = (min(L_tps),mini+(ajouty*2)))

    plt.annotate(variance,xy = (min(L_tps)+ajoutx ,mini+(ajouty*3)))
    plt.annotate('variance =',xy = (min(L_tps),mini+(ajouty*3)))

    plt.annotate(ecarttype,xy = (min(L_tps)+ajoutx ,mini+(ajouty*4)))
    plt.annotate('ecart type =',xy = (min(L_tps),mini+(ajouty*4)))

    plt.annotate(mediane,xy = (min(L_tps)+ajoutx ,mini+(ajouty*5)))
    plt.annotate('mediane =',xy = (min(L_tps),mini+(ajouty*5)))


    plt.show()








def detect_anomalie(num_capteur,donnee):
    k = 0
    liste_capteur = tableau[num_capteur-1]
    tps = []
    data = []
    L_tps = []
    erreur = True


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
            plt.plot([ano[0],ano[1]],[ano[3],ano[2]],'r-',label="arrêt du fonctionnement du capteur")
            plt.legend()
    else :
        erreur = True

    moy = moyenne_liste(data)
    ecart_typ = variance_liste(data)**0.5
    for i in range (len(data)-1) :
        if data[i+1] > moy + 3*ecart_typ or  data[i+1] < moy - 3*ecart_typ:
            erreur = True
            plt.plot([L_tps[i],L_tps[i+1]],[data[i],data[i+1]],'y-')

    plt.plot([0],[moy],'y-',label = "valeurs pouvant être considérées comme abérantes")
    plt.legend()

    if erreur == False :
        print ("Il n'y a pas d'anomalie")
    plt.ylabel(donnee)
    plt.xlabel('jours depuis le premier jour de mesure')
    plt.show()








def heure_nouveau(List1,a):

    if a == "noise" :
        c = 0
    elif a == "temp":
        c = 1
    elif a == "humidity":
        c = 2
    elif a == "lum" :
        c = 3
    elif a == "co2":
        c = 4
    else : print("Le nom de la variable que vous avez entrez n'existe pas")

    List=[]

    
    for i in range (len(List1)):
        if List1[i]==5: 
            if i!=0 : 
                List1[0],List1[i]=List1[i],List1[0]

    for i in List1:
        List.append(i-1)


    sent_at_nv,donnee_nv,nouvelle_heure,seconds_nv=[],[],[],[]
    k,w,x,y,z=0,0,0,0,0
    donnee_k0,donnee_k1,donnee_k2,donnee_k3,donnee_k4,donnee_k5=[],[],[],[],[],[]

    assert len(List)<7,"Il est demandé plus de courbe qu'il n'en existe"
    for index in range(len(List)) :
        assert List[index]==0 or List[index]==1 or List[index]==2 or List[index]==3 or List[index]==4 or List[index]==5,"Les numéros des capteurs demandées n'existe pas"
        for j in range (index):
            assert List[j]!=List[index], "il est demandé deux fois le même capteur"

                
    for index in List:

        sent_at_index,donnee_index,seconds_index=[],[],[]

        for i in tableau[index]:
            essai = datetime.datetime.strptime(i[5][:19], "%Y-%m-%d %H:%M:%S")
            essai_tuple=essai.timetuple()
            essai_seconds = calendar.timegm(essai_tuple)
            seconds_index.append(essai_seconds)

            sent_at_index.append(i[5])

            donnee_index.append(float(i[c]))

        sent_at_nv.append(sent_at_index)
        donnee_nv.append(donnee_index)
        seconds_nv.append(seconds_index)


    if len(List)==1:

        d='capteur'+str(List1[0])
        sent_at_nv[0]=def_time(sent_at_nv[0])
        plt.figure()
        plt.plot(sent_at_nv[0],donnee_nv[0],label=d)
        plt.legend()
        plt.ylabel(a)
        plt.xlabel('date')
        plt.show()



    elif len(List)==2:

        for i in range(len(sent_at_nv[0])-1):
            while seconds_nv[0][i]>=seconds_nv[1][k] and k<(len(sent_at_nv[1])-2):

                nouvelle_heure.append(sent_at_nv[1][k])

                donnee_k0.append(donnee_nv[0][i])
                donnee_k1.append(donnee_nv[1][k])

                k+=1

            nouvelle_heure.append(sent_at_nv[0][i])


            donnee_k1.append(donnee_nv[1][k])
            donnee_k0.append()

        d='capteur'+str(List1[0])
        d=str(d)
        e='capteur'+str(List1[1])
        e=str(e)
        
        nouvelle_heure=def_time(nouvelle_heure)
        
        plt.figure()
        plt.plot(nouvelle_heure,donnee_k0,label=d)
        plt.plot(nouvelle_heure,donnee_k1,label=e)
        plt.legend()
        plt.ylabel(a)
        plt.xlabel('date')
        plt.show()



    elif len(List)==3:

        for i in range(len(sent_at_nv[0])-1):
            while seconds_nv[0][i]>=seconds_nv[1][k] and k<(len(sent_at_nv[1])-2):
                while seconds_nv[1][k]>=seconds_nv[2][w] and w<(len(sent_at_nv[2])-1):

                    nouvelle_heure.append(sent_at_nv[2][w])
                    
                    donnee_k0.append(donnee_nv[0][i])
                    donnee_k1.append(donnee_nv[1][k])
                    donnee_k2.append(donnee_nv[2][w])

                    w+=1

                nouvelle_heure.append(sent_at_nv[1][k])

                donnee_k0.append(donnee_nv[0][i])
                donnee_k1.append(donnee_nv[1][k])
                donnee_k2.append(donnee_nv[2][w])

                k+=1

            nouvelle_heure.append(sent_at_nv[0][i])

            donnee_k0.append(donnee_nv[0][i])
            donnee_k1.append(donnee_nv[1][k])
            donnee_k2.append(donnee_nv[2][w])


        for i in range(len(seconds_nv)-1):
            if seconds_nv[i]==seconds_nv[i+1]:
                seconds_nv.pop(i)
                donnee_k0.pop(i)
                donnee_k1.pop(i)
                donnee_k2.pop(i)


        d='capteur'+str(List1[0])
        d=str(d)
        e='capteur'+str(List1[1])
        e=str(e)
        f='capteur'+str(List1[2])
        f=str(f)
        
        nouvelle_heure=def_time(nouvelle_heure)

        plt.figure()
        plt.plot(nouvelle_heure,donnee_k0,label=d)
        plt.plot(nouvelle_heure,donnee_k1,label=e)
        plt.plot(nouvelle_heure,donnee_k2,label=f)
        plt.legend()
        plt.ylabel(a)
        plt.xlabel('date')
        plt.show()


    elif len(List)==4 :

       for i in range(len(sent_at_nv[0])-1):
           while seconds_nv[0][i]>=seconds_nv[1][k] and k<(len(sent_at_nv[1])-1):
               while seconds_nv[1][k]>=seconds_nv[2][w] and w<(len(sent_at_nv[2])-1):

                   while seconds_nv[2][w]>=seconds_nv[3][x] and x<(len(sent_at_nv[3])-1):

                       nouvelle_heure.append(sent_at_nv[3][x])
                    
                       donnee_k0.append(donnee_nv[0][i])
                       donnee_k1.append(donnee_nv[1][k])
                       donnee_k2.append(donnee_nv[2][w])
                       donnee_k3.append(donnee_nv[3][x])

                       x+=1

                   nouvelle_heure.append(sent_at_nv[2][w])
                
                   donnee_k0.append(donnee_nv[0][i])
                   donnee_k1.append(donnee_nv[1][k])
                   donnee_k2.append(donnee_nv[2][w])
                   donnee_k3.append(donnee_nv[3][x])

                   w+=1

               nouvelle_heure.append(sent_at_nv[1][k])

               donnee_k0.append(donnee_nv[0][i])
               donnee_k1.append(donnee_nv[1][k])
               donnee_k2.append(donnee_nv[2][w])
               donnee_k3.append(donnee_nv[3][x])

               k+=1

           nouvelle_heure.append(sent_at_nv[0][i])
        
           donnee_k0.append(donnee_nv[0][i])
           donnee_k1.append(donnee_nv[1][k])
           donnee_k2.append(donnee_nv[2][w])
           donnee_k3.append(donnee_nv[3][x])



       d='capteur'+str(List1[0])
       d=str(d)
       e='capteur'+str(List1[1])
       e=str(e)
       f='capteur'+str(List1[2])
       f=str(f)
       g='capteur'+str(List1[3])
       g=str(g)
        
       nouvelle_heure=def_time(nouvelle_heure)

       plt.figure()
       plt.plot(nouvelle_heure,donnee_k0,label=d)
       plt.plot(nouvelle_heure,donnee_k1,label=e)
       plt.plot(nouvelle_heure,donnee_k2,label=f)
       plt.plot(nouvelle_heure,donnee_k3,label=g)
       plt.legend()
       plt.ylabel(a)
       plt.xlabel('date')
       plt.show()


    elif len(List)==5:

        for i in range(len(sent_at_nv[0])-1):
            while seconds_nv[0][i]>=seconds_nv[1][k] and k<(len(sent_at_nv[1])-1):
                while seconds_nv[1][k]>=seconds_nv[2][w] and w<(len(sent_at_nv[2])-1):
                     while seconds_nv[2][w]>=seconds_nv[3][x] and x<(len(sent_at_nv[3])-1):
                         while seconds_nv[3][x]>=seconds_nv[4][y] and y<(len(sent_at_nv[4])-1):

                             nouvelle_heure.append(sent_at_nv[4][y])
                             
                             donnee_k0.append(donnee_nv[0][i])
                             donnee_k1.append(donnee_nv[1][k])
                             donnee_k2.append(donnee_nv[2][w])
                             donnee_k3.append(donnee_nv[3][x])
                             donnee_k4.append(donnee_nv[4][y])

                             y+=1

                         nouvelle_heure.append(sent_at_nv[3][x])
                         
                         donnee_k0.append(donnee_nv[0][i])
                         donnee_k1.append(donnee_nv[1][k])
                         donnee_k2.append(donnee_nv[2][w])
                         donnee_k3.append(donnee_nv[3][x])
                         donnee_k4.append(donnee_nv[4][y])

                         x+=1

                     nouvelle_heure.append(sent_at_nv[2][w])

                     donnee_k0.append(donnee_nv[0][i])
                     donnee_k1.append(donnee_nv[1][k])
                     donnee_k2.append(donnee_nv[2][w])
                     donnee_k3.append(donnee_nv[3][x])
                     donnee_k4.append(donnee_nv[4][y])

                     w+=1

                nouvelle_heure.append(sent_at_nv[1][k])

                donnee_k0.append(donnee_nv[0][i])
                donnee_k1.append(donnee_nv[1][k])
                donnee_k2.append(donnee_nv[2][w])
                donnee_k3.append(donnee_nv[3][x])
                donnee_k4.append(donnee_nv[4][y])

                k+=1

            nouvelle_heure.append(sent_at_nv[0][i])

            donnee_k0.append(donnee_nv[0][i])
            donnee_k1.append(donnee_nv[1][k])
            donnee_k2.append(donnee_nv[2][w])
            donnee_k3.append(donnee_nv[3][x])
            donnee_k4.append(donnee_nv[4][y])


        d='capteur'+str(List1[0])
        d=str(d)
        e='capteur'+str(List1[1])
        e=str(e)
        f='capteur'+str(List1[2])
        f=str(f)
        g='capteur'+str(List1[3])
        g=str(g)
        h='capteur'+str(List1[4])
        h=str(h)

        nouvelle_heure=def_time(nouvelle_heure)

        plt.figure()
        plt.plot(nouvelle_heure,donnee_k0,label=d)
        plt.plot(nouvelle_heure,donnee_k1,label=e)
        plt.plot(nouvelle_heure,donnee_k2,label=f)
        plt.plot(nouvelle_heure,donnee_k3,label=g)
        plt.plot(nouvelle_heure,donnee_k4,label=h)
        plt.legend()
        plt.ylabel(a)
        plt.xlabel('date')
        plt.show()


    elif len(List)==6:


        for i in range(len(sent_at_nv[0])-1):
            while seconds_nv[0][i]>=seconds_nv[1][k] and k<(len(sent_at_nv[1])-1):
                while seconds_nv[1][k]>=seconds_nv[2][w] and w<(len(sent_at_nv[2])-1):
                     while seconds_nv[2][w]>=seconds_nv[3][x] and x<(len(sent_at_nv[3])-1):
                         while seconds_nv[3][x]>=seconds_nv[4][y] and y<(len(sent_at_nv[4])-1):
                             while seconds_nv[4][y]>=seconds_nv[5][z] and z<(len(sent_at_nv[5])-1):

                                 nouvelle_heure.append(sent_at_nv[5][z])

                                 donnee_k0.append(donnee_nv[0][i])
                                 donnee_k1.append(donnee_nv[1][k])
                                 donnee_k2.append(donnee_nv[2][w])
                                 donnee_k3.append(donnee_nv[3][x])
                                 donnee_k4.append(donnee_nv[4][y])
                                 donnee_k5.append(donnee_nv[5][z])

                                 z+=1

                             nouvelle_heure.append(sent_at_nv[4][y])

                             donnee_k0.append(donnee_nv[0][i])
                             donnee_k1.append(donnee_nv[1][k])
                             donnee_k2.append(donnee_nv[2][w])
                             donnee_k3.append(donnee_nv[3][x])
                             donnee_k4.append(donnee_nv[4][y])
                             donnee_k5.append(donnee_nv[5][z])

                             y+=1

                         nouvelle_heure.append(sent_at_nv[3][x])

                         donnee_k0.append(donnee_nv[0][i])
                         donnee_k1.append(donnee_nv[1][k])
                         donnee_k2.append(donnee_nv[2][w])
                         donnee_k3.append(donnee_nv[3][x])
                         donnee_k4.append(donnee_nv[4][y])
                         donnee_k5.append(donnee_nv[5][z])

                         x+=1

                     nouvelle_heure.append(sent_at_nv[2][w])

                     donnee_k0.append(donnee_nv[0][i])
                     donnee_k1.append(donnee_nv[1][k])
                     donnee_k2.append(donnee_nv[2][w])
                     donnee_k3.append(donnee_nv[3][x])
                     donnee_k4.append(donnee_nv[4][y])
                     donnee_k5.append(donnee_nv[5][z])

                     w+=1

                nouvelle_heure.append(sent_at_nv[1][k])

                donnee_k0.append(donnee_nv[0][i])
                donnee_k1.append(donnee_nv[1][k])
                donnee_k2.append(donnee_nv[2][w])
                donnee_k3.append(donnee_nv[3][x])
                donnee_k4.append(donnee_nv[4][y])
                donnee_k5.append(donnee_nv[5][z])

                k+=1

            nouvelle_heure.append(sent_at_nv[0][i])

            donnee_k0.append(donnee_nv[0][i])
            donnee_k1.append(donnee_nv[1][k])
            donnee_k2.append(donnee_nv[2][w])
            donnee_k3.append(donnee_nv[3][x])
            donnee_k4.append(donnee_nv[4][y])
            donnee_k5.append(donnee_nv[5][z])


        d='capteur'+str(List1[0])
        d=str(d)
        e='capteur'+str(List1[1])
        e=str(e)
        f='capteur'+str(List1[2])
        f=str(f)
        g='capteur'+str(List1[3])
        g=str(g)
        h='capteur'+str(List1[4])
        h=str(h)
        j='capteur'+str(List1[5])
        j=str(j)

        nouvelle_heure=def_time(nouvelle_heure)


        plt.figure()
        plt.plot(nouvelle_heure,donnee_k0,label=d)
        plt.plot(nouvelle_heure,donnee_k1,label=e)
        plt.plot(nouvelle_heure,donnee_k2,label=f)
        plt.plot(nouvelle_heure,donnee_k3,label=g)
        plt.plot(nouvelle_heure,donnee_k4,label=h)
        plt.plot(nouvelle_heure,donnee_k5,label=j)
        plt.legend()
        plt.ylabel(a)
        plt.xlabel('date')
        plt.show()






##

print ("Bonjour")
print ("Vous pouvez soit : \n 1. Tracer une courbe entre deux dates \n 2. Detecter les anomalies \n 3. Visualiser la corrélation entre deux variables \n 4. Visualiser la corrélation entre plusieurs capteurs \n 5. Afficher toutes les veleurs remarquables de la base de donnée")
print ("Donnez le numéro de votre choix")
choix = input()

if choix == "1" :
    print("Choisissez une donnée à tracer : temp, humidity, co2, noise, lum, humidex")
    choix_donnee = input()
    print("Choisissez un capteur : 1, 2, 3, 4, 5")
    choix_num = int(input())
    print("Choisissez une date de début sous la forme YYYY-MM-DD entre le 2019-08-11 et le 2019-08-25")
    date_debut = str(input())
    print("Choisissez une date de fin sous la forme YYYY-MM-DD entre le 2019-08-11 et le 2019-08-25")
    date_fin = str(input())
    trace(choix_num,choix_donnee,date_debut,date_fin)

if choix == "2" :
    print("Les anomalies détectées vont être tracées en rouge")
    print("Choisissez un capteur a diagnostiquer : 1, 2, 3, 4, 5")
    choix_num = int(input())
    print("Choisissez une donnée à tracer : temp, humidity, co2, noise, lum")
    choix_donnee = input()
    detect_anomalie(choix_num,choix_donnee)

if choix == "3" :
    print("Choisissez un capteur sur lequel visualiser la corrélation")
    choix_num = int(input())
    print("Choisissez une première donnée")
    premiere_donnee = input()
    print("Choisissez une seconde donnée")
    seconde_donnee = input()
    print ("Les données sont correlées à hauteur de : ", abs(correlation(choix_num,premiere_donnee,seconde_donnee))*100,"%")


if choix=="4":
    print("Choisissez une variable sur laquelle visualiser la corrélation")
    choix_variable = input()
    print("Choisissez le nombre de capteur à correler")
    nombre_capteur = int(input())
    assert nombre_capteur<7, "Vous demandez de correler plus de capteurs qu'il n'en existe"
    assert nombre_capteur>0, "Vous demandez un nombre de capteurs à corréler nul ou négatif"
    if nombre_capteur==1 :
        print("Choisissez le numéro du capteur")
        List1=[int(input())]
    elif nombre_capteur==2:
        print("Choisissez les numéros des capteurs à corréler")
        List1=[int(input()),int(input())]
    elif nombre_capteur==3:
        print("Choisissez les numéros des capteurs à corréler")
        List1=[int(input()),int(input()),int(input())]
    elif nombre_capteur==4:
        print("Choisissez les numéros des capteurs à corréler")
        List1=[int(input()),int(input()),int(input()),int(input())]
    elif nombre_capteur==5:
        print("Choisissez les numéros des capteurs à corréler")
        List1=[int(input()),int(input()),int(input()),int(input()),int(input())]
    elif nombre_capteur==6:
        List1=[5,1,2,3,4,6]

    heure_nouveau(List1,choix_variable)

if choix == "5" :
    all_stats()


##
