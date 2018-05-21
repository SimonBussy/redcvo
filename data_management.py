# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta


def get_data():
    # Load json data
    with open('../data/json_file.json') as data_file:
        patients = json.load(data_file)
    print("JSON file loaded")

    # Features computation
    print("Features computation launched...")
    visits = []
    for patient in patients.values():
        for i in range(1, len(patient['visits']) + 1):
            visits.append(patient['visits'][str(i)])

    n_visits = len(visits)
    print("n_visits = %s" % n_visits)

    # Features DataFrame with encounter_nums index
    encounter_nums = [int(visit.get('encounter_num')) for visit in visits]
    X = pd.DataFrame(index=encounter_nums)

    # Time vector & censoring indicator
    print("Adding labels...", end="")
    next_visit = [visit.get('next_visit') for visit in visits]

    T = np.array([1e10 if str(t) == 'none' else t for t in next_visit]).astype(
        int)
    end_dates = pd.to_datetime([visit.get('end_date') for visit in visits])
    start_dates = pd.to_datetime([visit.get('start_date') for visit in visits])
    C = pd.to_datetime('2016-01-15 00:00:00') - end_dates
    days, seconds = C.days, C.seconds
    C = days * 24 + seconds // 3600  # in hours (discrete)
    delta = (T <= C).astype(int)
    Y = T
    Y[delta == 0] = C[delta == 0]
    labels = pd.DataFrame({'Y': Y, 'delta': delta}, index=encounter_nums)

    X = pd.concat([X, labels], axis=1)
    print("Done!")

    # Basic features
    print("Adding basic features...", end="")
    # Add also patient_num & encounter_num for future random choice
    patient_num, encounter_num = [], []
    sex, baseline_HB, genotype_SS, age, transfu_count = [], [], [], [], []
    LS_ALONE, LS_INACTIVE, MH_ACS, MH_AVN, MH_DIALISIS = [], [], [], [], []
    MH_HEART_FAILURE, MH_ISCHEMIC_STROKE, MH_LEG_ULCER = [], [], []
    MH_NEPHROPATHY, MH_PHTN, MH_PRIAPISM, MH_RETINOPATHY = [], [], [], []
    OPIOID_TO_DISCHARGE, ORAL_OPIOID, USED_MORPHINE = [], [], []
    USED_OXYCODONE, duration, previous_visit, rea = [], [], [], []

    for patient in patients.values():
        for _ in range(1, len(patient['visits']) + 1):
            patient_num.append(patient['patient_num'])
            sex.append(1 if int(patient['sex']) == 1 else 0)
            baseline_HB.append(patient['baseline_HB'])
            genotype_SS.append(patient['genotype_SS'])

    for visit in visits:
        encounter_num.append(visit.get('encounter_num'))
        age.append(visit.get('age'))
        rea.append(visit.get('rea'))
        LS_ALONE.append(visit.get('LS_ALONE'))
        LS_INACTIVE.append(visit.get('LS_INACTIVE'))
        MH_ACS.append(visit.get('MH_ACS'))
        MH_AVN.append(visit.get('MH_AVN'))
        MH_DIALISIS.append(visit.get('MH_DIALISIS'))
        MH_HEART_FAILURE.append(visit.get('MH_HEART_FAILURE'))
        MH_ISCHEMIC_STROKE.append(visit.get('MH_ISCHEMIC_STROKE'))
        MH_LEG_ULCER.append(visit.get('MH_LEG_ULCER'))
        MH_NEPHROPATHY.append(visit.get('MH_NEPHROPATHY'))
        MH_PHTN.append(visit.get('MH_PHTN'))
        MH_PRIAPISM.append(visit.get('MH_PRIAPISM'))
        MH_RETINOPATHY.append(visit.get('MH_RETINOPATHY'))
        ORAL_OPIOID.append(visit.get('ORAL_OPIOID'))
        USED_MORPHINE.append(visit.get('USED_MORPHINE'))
        USED_OXYCODONE.append(visit.get('USED_OXYCODONE'))
        duration.append(visit.get('duration'))
        transfu_count.append(visit.get('transfu_count'))

    MH_ACS = [1 if int(x) == 2 else x for x in MH_ACS]
    MH_AVN = [1 if int(x) == 2 else x for x in MH_AVN]
    MH_DIALISIS = [1 if int(x) == 2 else x for x in MH_DIALISIS]
    MH_HEART_FAILURE = [1 if int(x) == 2 else x for x in MH_HEART_FAILURE]
    MH_ISCHEMIC_STROKE = [1 if int(x) == 2 else x for x in MH_ISCHEMIC_STROKE]
    MH_LEG_ULCER = [1 if int(x) == 2 else x for x in MH_LEG_ULCER]
    MH_NEPHROPATHY = [1 if int(x) == 2 else x for x in MH_NEPHROPATHY]
    MH_PHTN = [1 if int(x) == 2 else x for x in MH_PHTN]
    MH_PRIAPISM = [1 if int(x) == 2 else x for x in MH_PRIAPISM]
    MH_RETINOPATHY = [1 if int(x) == 2 else x for x in MH_RETINOPATHY]

    X_basic = pd.DataFrame(
        {'patient_num': patient_num, 'encounter_num': encounter_num, 'sex': sex,
         'start_dates': start_dates, 'end_dates': end_dates,
         'genotype_SS': genotype_SS, 'age': age, 'rea': rea,
         'LS_INACTIVE': LS_INACTIVE, 'MH_ACS': MH_ACS, 'MH_AVN': MH_AVN,
         'MH_DIALISIS': MH_DIALISIS, 'MH_HEART_FAILURE': MH_HEART_FAILURE,
         'MH_ISCHEMIC_STROKE': MH_ISCHEMIC_STROKE,
         'MH_LEG_ULCER': MH_LEG_ULCER, 'LS_ALONE': LS_ALONE,
         'MH_NEPHROPATHY': MH_NEPHROPATHY, 'MH_PHTN': MH_PHTN,
         'MH_PRIAPISM': MH_PRIAPISM, 'MH_RETINOPATHY': MH_RETINOPATHY,
         'ORAL_OPIOID': ORAL_OPIOID, 'baseline_HB': baseline_HB,
         'USED_MORPHINE': USED_MORPHINE, 'USED_OXYCODONE': USED_OXYCODONE,
         'duration': duration, 'transfu_count': transfu_count},
        index=encounter_nums)

    X_basic = X_basic.convert_objects(convert_numeric=True)
    X = pd.concat([X, X_basic], axis=1)
    print("Done!")

    # Bio data
    print("Adding bio features...", end="")
    bio_data, bio_names = pd.DataFrame(), []
    for visit in visits:
        encounter_num = int(visit.get('encounter_num'))
        tmp = pd.DataFrame()
        tmp = tmp.append(pd.Series(), ignore_index=True)
        tmp.index = [encounter_num]
        for bio_name, bio_values in visit.get('bio').items():
            bio_names.append(bio_name)
            vals, times = [], []
            for val in bio_values.values():
                vals.append(val['nval_num']), times.append(val['date_bio'])
            tmp[bio_name] = np.empty
            tmp[bio_name][encounter_num] = pd.Series(vals, index=times)
        bio_data = bio_data.append(tmp)

    bio_names_count = pd.Series(
        bio_names).value_counts() * 100 / n_visits
    bio_percentage = 35  # 35% validated with Raph (13.03.2017)
    bio_param_kept = bio_names_count[bio_names_count > bio_percentage]
    bio_data = bio_data[bio_param_kept.index]
    print("Done!")

    X = pd.concat([X, bio_data], axis=1)

    # Vital parameters data
    print("Adding vital parameters features...")
    param_gp = ['Fréquence cardiaque [bpm]',
                'Fréquence respiratoire [mvt/min]', 'PA max [mmHg]',
                'PA min [mmHg]', 'Température [°C]',
                'Saturation en oxygène [%]']

    vital_parameter_data, vital_parameter_names = pd.DataFrame(), []
    for visit in visits:
        encounter_num = int(visit.get('encounter_num'))
        tmp = pd.DataFrame()
        tmp = tmp.append(pd.Series(), ignore_index=True)
        tmp.index = [encounter_num]
        for vital_name, vital_values in visit.get('vital_parameters').items():
            if vital_name in param_gp:
                vital_parameter_names.append(vital_name)
                vals, times = [], []
                for val in vital_values.values():
                    vals.append(val['nval_num']), times.append(
                        val['start_date'])
                outliers = False
                if (vital_name == 'Fréquence respiratoire [mvt/min]'
                    and np.count_nonzero(np.array(vals).astype(float) > 60) > 0) \
                        or (vital_name == 'Saturation en oxygène [%]'
                            and (np.count_nonzero(
                                    np.array(vals).astype(float) > 150) > 0
                                 or np.count_nonzero(np.array(vals).astype(
                                    float) < 50) > 0)) \
                        or (vital_name == 'Température [°C]'
                            and np.count_nonzero(
                                    np.array(vals).astype(float) < 30) > 0):
                    outliers = True
                if not outliers:
                    tmp[vital_name] = np.empty
                    serie = pd.Series(vals, index=times).astype(float)

                    # align times on 8AM
                    pt1_datetime = datetime.strptime(serie.index[0],
                                                     '%Y-%m-%d  %H:%M:%S')
                    same_day_8h = str(pt1_datetime.date()) + '  08:00:00'
                    same_day_8h_datetime = datetime.strptime(same_day_8h,
                                                             '%Y-%m-%d  %H:%M:%S')
                    day_before_8h_datetime = same_day_8h_datetime - timedelta(
                        days=1)
                    if same_day_8h_datetime > pt1_datetime:
                        align_on = day_before_8h_datetime
                    else:
                        align_on = same_day_8h_datetime
                    serie = pd.concat(
                        [pd.Series(index=[str(align_on)], data=np.nan), serie])

                    tmp[vital_name][encounter_num] = serie
        vital_parameter_data = vital_parameter_data.append(tmp)
    print("Done!")

    X = pd.concat([X, vital_parameter_data], axis=1)

    # drop outliers
    X = X[X["encounter_num"] != 2008581722]
    X = X[X["encounter_num"] != 2009640859]
    X = X[X["encounter_num"] != 2011819004]
    X = X[X["encounter_num"] != 2008734750]

    # drop pb encounter_num
    X = X[X["encounter_num"] != 2011748771]

    # drop non-sens columns (% version of covariates)
    X = X.drop([' 04. (02PNP) Polynucleaires neutrophiles'], axis=1)
    X = X.drop([' 11. (02LYP) Lymphocytes'], axis=1)
    X = X.drop([' 13. (02MOP) Monocytes'], axis=1)
    X = X.drop([' 09. (02PBP) Polynucleaires basophiles'], axis=1)
    X = X.drop([' 06. (02PEP) Polynucleaires eosinophiles'], axis=1)

    # Rename Columns
    current_names = [
        'LS_ALONE', 'LS_INACTIVE', 'MH_ACS', 'MH_AVN', 'MH_DIALISIS',
        'MH_HEART_FAILURE', 'MH_ISCHEMIC_STROKE', 'MH_LEG_ULCER',
        'MH_NEPHROPATHY', 'MH_PHTN', 'MH_PRIAPISM', 'MH_RETINOPATHY',
        'OPIOID_TO_DISCHARGE', 'ORAL_OPIOID', 'USED_MORPHINE', 'USED_OXYCODONE',
        'age', 'baseline_HB', 'duration', 'genotype_SS',
        'previous_visit', 'rea', 'sex', ' 05. (02HT) Hématocrite',
        ' 03. (02GR) Hématies', ' 07. (02TGMH) Teneur Globulaire Moyenne',
        ' 04. (02HB) Hémoglobine', ' 02. (02GB) Leucocytes',
        ' 06. (02VGM) Volume Globulaire Moyen',
        ' 04. (02PNP) Polynucleaires neutrophiles',
        'Nb of polynucleaires eosinophiles', 'Nb of lymphocytes',
        'Nb of polynucleaires neutrophiles', 'Nb of monocytes',
        ' 11. (02LYP) Lymphocytes', ' 13. (02MOP) Monocytes',
        ' 09. (02PBP) Polynucleaires basophiles',
        'Nb of polynucleaires basophiles', ' 16. (01CL) Chlorures',
        ' 18. (01CO2) CO2 Total', ' 14. (01NA) Sodium', ' 19. (01PT) Protéines',
        ' 24. (01DFG1) DFG estimé (MDRD)', ' 22. (01CRE1) Créatinine',
        ' 15. (01K) Potassium', ' 3. (02VPM) Volume Plaquettaire Moyen',
        ' 11. (02RETAB) Réticulocytes', ' 49. (01BT) Bilirubine totale',
        ' 47. (01PAL) Phosphatases alcalines', ' 45. (01ASAT) ASAT',
        ' 10. (02PL) Plaquettes', ' 46. (01ALAT) ALAT',
        ' 06. (02PEP) Polynucleaires eosinophiles',
        ' 02. (01CRP) Protéine C-reactive', ' 48. (01GGT) Gamma GT',
        ' 50. (01BC) Bilirubine conjuguée', ' 20. (01CA) Calcium',
        ' 28. (01G) Glucose', ' 39. (02ERY) Erythroblastes',
        ' 41. (01LDH1) Lactate Deshydrogénase',
        ' 08. (02CCMH) Concentration corpusculaire moyenne en hémoglobine',
        ' 08. (01HBS) Hémoglobine S', ' 06. (01HBF) Hémoglobine F',
        ' 21. (01U) Urée', 'Débit O2 [L/min] delay',
        'Fréquence cardiaque [bpm]',
        'Fréquence cardiaque [bpm] cst_kernel',
        'Fréquence cardiaque [bpm] mean',
        'Fréquence cardiaque [bpm] length_scale_RBF',
        'Fréquence cardiaque [bpm] noise_level',
        'Fréquence cardiaque [bpm] slope',
        'Saturation en oxygène [%]',
        'Saturation en oxygène [%] cst_kernel',
        'Saturation en oxygène [%] mean',
        'Saturation en oxygène [%] length_scale_RBF',
        'Saturation en oxygène [%] noise_level',
        'Saturation en oxygène [%] slope',
        'Fréquence respiratoire [mvt/min]',
        'Fréquence respiratoire [mvt/min] cst_kernel',
        'Fréquence respiratoire [mvt/min] mean',
        'Fréquence respiratoire [mvt/min] length_scale_RBF',
        'Fréquence respiratoire [mvt/min] noise_level',
        'Fréquence respiratoire [mvt/min] slope',
        'PA max [mmHg]',
        'PA max [mmHg] cst_kernel',
        'PA max [mmHg] mean', 'PA max [mmHg] length_scale_RBF',
        'PA max [mmHg] noise_level', 'PA max [mmHg] slope',
        'PA min [mmHg]',
        'PA min [mmHg] cst_kernel', 'PA min [mmHg] mean',
        'PA min [mmHg] length_scale_RBF', 'PA min [mmHg] noise_level',
        'PA min [mmHg] slope', 'Poids [kg] mean', 'Taille [cm] mean',
        'Température [°C]',
        'Température [°C] cst_kernel', 'Température [°C] mean',
        'Température [°C] length_scale_RBF', 'Température [°C] noise_level',
        'Température [°C] slope', 'BMI', 'area bolus dosage', 'area max dosage',
        'area refactory period', 'slope bolus dosage', 'mean bolus dosage',
        'slope max dosage', 'mean max dosage', 'slope refactory period',
        'mean refactory period', 'slope delay between syringe',
        'syringe frequency', 'transfu_count', ' 28. (01TCO2) CO2 Total',
        ' 30. (01HBGS) Hemoglobine', ' 27. (01SAO2) SaO2', ' 24. (01PH) pH',
        ' 32. (01HTEGS) Hematocrite', ' 26. (01PCO2) pCO2',
        ' 25. (01PO2) pO2', ' 14. (01TEMP) Temperature',
        ' 29. (01EB) Exces de bases', ' 3. (02TP%) TP', ' 4. (02INR) INR']

    new_names = [
        'Household situation', 'Professional activity',
        'History of acute chest syndrom', 'History of avascular bone necrosis',
        'Formerly or currently on a dialysis protocol',
        'History of heart failure', 'History of ischemic stroke',
        'History of leg skin ulceration', 'History of known nephropathy',
        'History of pulmonary hypertension', 'History of priapism*',
        'History of known retinopathy',
        'Post-opioid observation period (hours)',
        'Received orally administered opioids', 'Received Morphine',
        'Received Oxycodone', 'Age at hospital admission',
        'Baseline haemoglobin ($g/dL$)', 'Length of hospital stay',
        'Genotype', 'Less than 18 months since last visit', 'Stayed in ICU',
        'Gender', 'Hematocrit ($\%$)', 'Red blood cells ($10^{12}/L$)',
        'Mean corpuscular hemoglobin ($pg$)',
        'Hemoglobin ($g/dL$)', 'White blood cells ($10^9/L$)',
        'Mean cell volume ($fl$)',
        'Neutrophils ($\%$)', 'Eosinophils ($10^9/L$)',
        'Lymphocytes ($10^9/L$)', 'Neutrophils ($10^9/L$)',
        'Monocytes ($10^9/L$)', 'Lymphocytes ($\%$)', 'Monocytes ($\%$)',
        'Basophils ($\%$)',
        'Basophils ($10^9/L$)', 'Chloride ($mmol/L$)', 'Bicarbonate ($mmol/L$)',
        'Sodium ($mmol/L$)', 'Proteins ($g/L$)',
        'Renal function by MDRD ($mL/min/1,73m2$)',
        'Creatinine ($\mu mol/L$)', 'Potassium ($mmol/L$)',
        'Mean platelet volume ($fl$)', 'Reticulocytes ($10^9/L$)',
        'Total bilirubin ($\mu mol/L$)', 'Alkaline phosphatase ($U/L$)',
        'Asparate transaminase ($U/L$)', 'Platelets ($10^9/L$)',
        'Alanine transaminase ($U/L$)', 'Eosinophils ($\%$)',
        'C-reactive protein ($mg/L$)',
        'Gamma glutamyl-tranferase ($U/L$)', 'Direct bilirubin ($\mu mol/L$)',
        'Total calcium ($mmol/L$)', 'Glucose ($mmol/L$)',
        'Nucleated red blood cells ($10^9/L$)', 'Lactate Dehydrogenase ($U/L$)',
        'Mean corpuscular hemoglobin concentration ($\%$)',
        'Hemoglobin S ($\%$)', 'Hemoglobin F ($\%$)', 'Urea ($mmol/L$)',
        'Post-oxygen observation period (hours)',
        'Heart rate (bpm)',
        'Heart rate (constant kernel)', 'Heart rate (average)',
        'Heart rate (radial basis function kernel)',
        'Heart rate (noise level kernel)', 'Heart rate (slope)',
        'Oxygen saturation ($\%$)',
        'Oxygen saturation (constant kernel)', 'Oxygen saturation (average)',
        'Oxygen saturation (radial basis function kernel)',
        'Oxygen saturation (noise level kernel)', 'Oxygen saturation (slope)',
        'Respiratory rate (mvt/min)',
        'Respiratory rate (constant kernel)', 'Respiratory rate (average)',
        'Respiratory rate (radial basis function kernel)',
        'Respiratory rate (noise)', 'Respiratory rate (slope)',
        'Systolic blood pressure ($mmHg$)',
        'Systolic blood pressure (constant kernel)',
        'Systolic blood pressure (average)',
        'Systolic blood pressure (radial basis function kernel)',
        'Systolic blood pressure (noise level kernel)',
        'Systolic blood pressure (slope)',
        'Diastolic blood pressure ($mmHg$)',
        'Diastolic blood pressure (constant kernel)',
        'Diastolic blood pressure (average)',
        'Diastolic blood pressure (radial basis function kernel)',
        'Diastolic blood pressure (noise level kernel)',
        'Diastolic blood pressure (slope)', 'Weight ($kg$)', 'Size ($cm$)',
        'Temperature ($^\circ C$)',
        'Temperature (constant kernel)', 'Temperature (average)',
        'Temperature (radial basis function kernel)',
        'Temperature (noise level kernel)', 'Temperature (slope)',
        'Body mass index ($kg/m^2$)', 'Bolus dosage (area)',
        'Maximum dosage (area)', 'Refractory period (area)',
        'Bolus dosage (slope)', 'Bolus dosage (average)',
        'Maximum dosage (slope)', 'Maximum dosage (average)',
        'Refractory period (slope)', 'Refractory period (average)',
        'Delay between syringes (slope)', 'Syringe frequency (per day)',
        'Transfusion count', 'ABG: total CO2 (mmol/L)',
        'ABG: hemoglobin (g/dL)', 'ABG: oxygen saturation (percent)', 'ABG: pH',
        'ABG: hematocrit (percent)', 'ABG: oxygen partial pressure (mmHg)',
        'ABG: carbon dioxide partial pressure (mmHg)',
        'ABG: temperature (celcius)', 'ABG: base excess (mmol/L)',
        'Prothrombin Ratio (percent)', 'International Normalized Ratio']

    renamed_columns = list()
    for name in X.columns:
        idx = [i for i, x in enumerate(current_names) if x == name]
        try:
            renamed_columns.append(new_names[idx[0]])
        except:
            renamed_columns.append(name)

    X.columns = renamed_columns
    X.encounter_num = X.encounter_num.convert_objects(convert_numeric=True)
    X.patient_num = X.patient_num.convert_objects(convert_numeric=True)

    # drop ABG covariates (not enough points)
    X = X.drop(['ABG: total CO2 (mmol/L)'], axis=1)
    X = X.drop(['ABG: hemoglobin (g/dL)'], axis=1)
    X = X.drop(['ABG: oxygen saturation (percent)'], axis=1)
    X = X.drop(['ABG: pH'], axis=1)
    X = X.drop(['ABG: hematocrit (percent)'], axis=1)
    X = X.drop(['ABG: oxygen partial pressure (mmHg)'], axis=1)
    X = X.drop(['ABG: carbon dioxide partial pressure (mmHg)'], axis=1)
    X = X.drop(['ABG: temperature (celcius)'], axis=1)
    X = X.drop(['ABG: base excess (mmol/L)'], axis=1)

    # Add GHM & ZIP infos
    GHM = pd.read_csv("../data/GHM.csv", sep=";")
    GHM.CONCEPT_CD = GHM.CONCEPT_CD.apply(
        lambda x: x[-1].replace('T', '1').replace('Z', '1'))

    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    encoded_ghm = one_hot_encoder.fit_transform(np.atleast_2d(GHM.CONCEPT_CD).T)
    encoded_ghm = pd.DataFrame(encoded_ghm,
                               columns=["GHM=1", "GHM=2",
                                        "GHM=3", "GHM=4"])
    GHM = pd.concat([GHM, encoded_ghm], axis=1)
    GHM = GHM.drop('CONCEPT_CD', 1)
    X = X.merge(GHM, how="left", on='encounter_num')

    ZIP = pd.read_csv("../data/ZIP.csv", sep=";")
    X = X.merge(ZIP.convert_objects(convert_numeric=True),
                how="left", on='patient_num')
    X.index = X.encounter_num
    X = X.drop(['encounter_num'], axis=1)

    return X
