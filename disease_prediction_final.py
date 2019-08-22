from tkinter import *
import numpy as np
import pandas as pd


l1=['itching', 'skin_rash','nodal_skin_eruptions','continuous_sneezing',
       'shivering', 'chills', 'joint_pain','stomach_pain','acidity',
       'ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination',
       'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness',
       'lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness',
       'sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite',
       'pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever',
       'yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
       'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',	
       'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain',	
       'weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region',	
       'bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity',	
       'swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid',	
       'brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts', 	
       'drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness',	
       'stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance',	
       'unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
       'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression',	
'irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation',	
'dischromic _patches','watering_from_eyes','increased_appetite',	'polyuria','family_history',	
'mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',	
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',	
'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples',
'blackheads','scurring', 'skin_peeling', 'silver_like_dusting',
       'small_dents_in_nails','inflammatory_nails','blister',
       'red_sore_around_nose','yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)
    

df=pd.read_csv("D:\Machine Learning\Disease Prediction\Disease\Training.csv")    

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X= df[l1]

y = df[["prognosis"]]

np.ravel(y)

tr=pd.read_csv("D:\Machine Learning\Disease Prediction\Disease\Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

#Prediction Using Decision tree algorithm


        

#Prediction using Random Forest Algorithm
        
def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


#Prediction Using Naive Bayes algorithm
def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")
        
        
#Predicting some results:
from sklearn import tree

clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
clf3 = clf3.fit(X,y)

# calculating accuracy-------------------------------------------------------------------
from sklearn.metrics import accuracy_score
y_pred=clf3.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred,normalize=False))
# -----------------------------------------------------

psymptoms = ['itching', 'skin_rash','nodal_skin_eruptions']

for k in range(0,len(l1)):
    # print (k,)
    for z in psymptoms:
        if(z==l1[k]):
            l2[k]=1

inputtest = [l2]
predict = clf3.predict(inputtest)
predicted=predict[0]

for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

print(disease[a])

dataframe_cure = pd.read_csv("D:\Machine Learning\Disease Prediction\Disease\cure_data.csv")
cures = dataframe_cure.herbal_treatment
cures2 = cures.tolist()

X_cure = dataframe_cure.drop('herbal_treatment',axis=1)
y_cure = dataframe_cure[['herbal_treatment']]

np.ravel(y_cure)

cureszero = []
for x in range(0,len(cures2)):
    cureszero.append(0)


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y_cure = labelencoder.fit_transform(y_cure)

from sklearn.tree import DecisionTreeClassifier
decsion_cure = DecisionTreeClassifier()
decsion_cure = decsion_cure.fit(X_cure,y_cure)

cure_disease = ['Common Cold']

for k in range(0,len(disease)):
    # print (k,)
    for z in cure_disease:
        if(z==disease[k]):
            cureszero[k]=1

inputtest1 = [cureszero]
predict1 = decsion_cure.predict(inputtest1)
predicted1=predict1[0]
print(predicted1)

for a in range(0,len(cures2)):
        if(predicted1 == y_cure[a]):
            break
        
print(cures2[a])

from sklearn.ensemble import RandomForestClassifier
algCure = RandomForestClassifier(n_estimators=500)






           


