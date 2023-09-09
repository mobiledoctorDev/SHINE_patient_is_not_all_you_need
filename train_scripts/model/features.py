
dict_set_of_cols = {
    'patient': [
        'cough', 'fever', 'sore_throat', 'shortness_of_breath', 'sputum',
        'head_ache', 'runny_nose', 'muscle_pain', 'chills',
        'loss_of_taste', 'loss_of_smell', 'chest_pain',
        'indication_other', 'indication_abroad', 'indication_contact',
        'gender', 'age_ratio',
    ],
    'patient2': [
        "cough", "sore_throat", "shortness_of_breath", "sputum",
        "head_ache", "runny_nose", "muscle_pain", "fever", "chills", "fatigue",
        "loss_appetite", "loss_of_taste", "loss_of_smell", "chest_pain", "diarrhea", "constipation",
        "red_eyes", "blue/pale_lips", "sneezing", "body_rash", "lip_blisters",
        "indication_other", "indication_abroad", "indication_contact",
        'gender', 'age_ratio',
    ],
    'screason': [
        'scReason1',
        'scReason2',
        'scReason3',
        'scReason4',
        'scReason5',
        'scReason6',
        'scReason7',
        'scReason8',
        'scReason9',
        'scReason10',
    ],
    'si10': [
        'weekday',
        'global_confirmed_ratio',
        'confirmed_ratio',
        'sigungu_confirmed_ratio',
        'sigungu_confirmed_mask',
        'total_cases_per_million_for_6months_norm',
        'total_cases_per_million_for_6months_mask',
        'total_vaccinations_per_hundred_for_6months_norm',
        'total_vaccinations_per_hundred_for_6months_mask',
        'total_deaths_per_million_for_6months_norm',
        'total_deaths_per_million_for_6months_mask',
        'reproduction_rate_norm',
        'reproduction_rate_mask',
        'positive_rate_norm',
        'positive_rate_mask',
        'weekly_hosp_admissions_per_million_norm',
        'weekly_hosp_admissions_per_million_mask',
        'people_fully_vaccinated_per_hundred_norm',
        'people_fully_vaccinated_per_hundred_mask',
    ],
    'hw7': [
        'weekday',
        'global_confirmed_ratio',
        'sigungu_confirmed_ratio',
        'sigungu_confirmed_mask',
        'new_cases_per_million_norm',
        'new_cases_per_million_mask',
        'reproduction_rate_norm',
        'reproduction_rate_mask',
        'positive_rate_norm',
        'positive_rate_mask',
        'weekly_hosp_admissions_per_million_norm',
        'weekly_hosp_admissions_per_million_mask',
        'people_fully_vaccinated_per_hundred_norm',
        'people_fully_vaccinated_per_hundred_mask',
    ],
    'owid': [
        'total_cases_per_million_norm',
        'total_cases_per_million_mask',
        'new_cases_per_million_norm',
        'new_cases_per_million_mask',
        'new_cases_smoothed_per_million_norm',
        'new_cases_smoothed_per_million_mask',
        'total_deaths_per_million_norm',
        'total_deaths_per_million_mask',
        'new_deaths_per_million_norm',
        'new_deaths_per_million_mask',
        'new_deaths_smoothed_per_million_norm',
        'new_deaths_smoothed_per_million_mask',
        'icu_patients_per_million_norm',
        'icu_patients_per_million_mask',
        'hosp_patients_per_million_norm',
        'hosp_patients_per_million_mask',
        'weekly_icu_admissions_per_million_norm',
        'weekly_icu_admissions_per_million_mask',
        'weekly_hosp_admissions_per_million_norm',
        'weekly_hosp_admissions_per_million_mask',
        'total_tests_per_thousand_norm',
        'total_tests_per_thousand_mask',
        'new_tests_per_thousand_norm',
        'new_tests_per_thousand_mask',
        'new_tests_smoothed_per_thousand_norm',
        'new_tests_smoothed_per_thousand_mask',
        'total_vaccinations_per_hundred_norm',
        'total_vaccinations_per_hundred_mask',
        'people_vaccinated_per_hundred_norm',
        'people_vaccinated_per_hundred_mask',
        'people_fully_vaccinated_per_hundred_norm',
        'people_fully_vaccinated_per_hundred_mask',
        'total_boosters_per_hundred_norm',
        'total_boosters_per_hundred_mask',
        'new_vaccinations_smoothed_per_million_norm',
        'new_vaccinations_smoothed_per_million_mask',
        'new_people_vaccinated_smoothed_per_hundred_norm',
        'new_people_vaccinated_smoothed_per_hundred_mask',
        'hospital_beds_per_thousand_norm',
        'hospital_beds_per_thousand_mask',
        'excess_mortality_cumulative_per_million_norm',
        'excess_mortality_cumulative_per_million_mask',
        'reproduction_rate_norm',
        'reproduction_rate_mask',
        'positive_rate_norm',
        'positive_rate_mask',
    ],
}

target_loc_cols_template = ['n_norm', 'n_mask', 'loc_std_norm', 'loc_std_mask', 'path_sum_norm', 'path_sum_mask']

gps_cols = [f"{col}_gps-{x}" for x in range(1, 8) for col in target_loc_cols_template]
dict_set_of_cols.update({
    'gps': gps_cols
})
bts_cols = [f"{col}_bts-{x}" for x in range(1, 8) for col in target_loc_cols_template]
dict_set_of_cols.update({
    'bts': bts_cols
})


def convert_gs_save_string(col):
    new_col = col.replace("-", '_')
    new_col = new_col.replace("+", 'plus')
    new_col = new_col.replace(" ", '_')
    new_col = new_col.replace("(", '_')
    new_col = new_col.replace(")", '_')
    new_col = new_col.replace("/", '_')
    new_col = new_col.replace(":", '_')
    new_col = new_col.replace(".", '_')
    new_col = new_col.replace(",", '_')
    new_col = new_col.replace("?", '_')
    new_col = new_col.replace("=", '_')
    new_col = new_col.replace(">", '_')
    new_col = new_col.replace("<", '_')
    return new_col


column_specs = {
    'age_ratio': 'numeric',
    'chest_pain': 'categorical',
    'chills': 'categorical',
    'cough': 'categorical',
    'fever': 'categorical',
    'gender': 'categorical',
    'head_ache': 'categorical',
    'indication_abroad': 'categorical',
    'indication_contact': 'categorical',
    'indication_other': 'categorical',
    'loss_of_smell': 'categorical',
    'loss_of_taste': 'categorical',
    'muscle_pain': 'categorical',
    'runny_nose': 'categorical',
    'shortness_of_breath': 'categorical',
    'sore_throat': 'categorical',
    'sputum': 'categorical',
    'scReason1': 'categorical',
    'scReason2': 'categorical',
    'scReason3': 'categorical',
    'scReason4': 'categorical',
    'scReason5': 'categorical',
    'scReason6': 'categorical',
    'scReason7': 'categorical',
    'scReason8': 'categorical',
    'scReason9': 'categorical',
    'scReason10': 'categorical',
    'n_norm_1': 'numeric',
    'loc_std_norm_1': 'numeric',
    'path_sum_norm_1': 'numeric',
    'n_norm_2': 'numeric',
    'loc_std_norm_2': 'numeric',
    'path_sum_norm_2': 'numeric',
    'n_norm_3': 'numeric',
    'loc_std_norm_3': 'numeric',
    'path_sum_norm_3': 'numeric',
    'n_norm_4': 'numeric',
    'loc_std_norm_4': 'numeric',
    'path_sum_norm_4': 'numeric',
    'n_norm_5': 'numeric',
    'loc_std_norm_5': 'numeric',
    'path_sum_norm_5': 'numeric',
    'n_norm_6': 'numeric',
    'loc_std_norm_6': 'numeric',
    'path_sum_norm_6': 'numeric',
    'n_norm_7': 'numeric',
    'loc_std_norm_7': 'numeric',
    'path_sum_norm_7': 'numeric',
    'weekday': 'categorical',
    'global_confirmed_ratio': 'numeric',
    'confirmed_ratio': 'numeric',
    'sigungu_confirmed_ratio': 'numeric',
    'total_cases_per_million_for_6months_norm': 'numeric',
    'total_vaccinations_per_hundred_for_6months_norm': 'numeric',
    'total_deaths_per_million_for_6months_norm': 'numeric',
    'total_cases_per_million_norm': 'numeric',
    'total_deaths_per_million_norm': 'numeric',
    'total_tests_per_thousand_norm': 'numeric',
    'total_vaccinations_per_hundred_norm': 'numeric',
    'total_boosters_per_hundred_norm': 'numeric',
    'new_cases_per_million_norm': 'numeric',
    'new_cases_smoothed_per_million_norm': 'numeric',
    'new_deaths_per_million_norm': 'numeric',
    'new_deaths_smoothed_per_million_norm': 'numeric',
    'new_tests_per_thousand_norm': 'numeric',
    'new_tests_smoothed_per_thousand_norm': 'numeric',
    'icu_patients_per_million_norm': 'numeric',
    'hosp_patients_per_million_norm': 'numeric',
    'hospital_beds_per_thousand_norm': 'numeric',
    'weekly_icu_admissions_per_million_norm': 'numeric',
    'weekly_hosp_admissions_per_million_norm': 'numeric',
    'people_vaccinated_per_hundred_norm': 'numeric',
    'people_fully_vaccinated_per_hundred_norm': 'numeric',
    'new_vaccinations_smoothed_per_million_norm': 'numeric',
    'new_people_vaccinated_smoothed_per_hundred_norm': 'numeric',
    'excess_mortality_cumulative_per_million_norm': 'numeric',
    'reproduction_rate_norm': 'numeric',
    'positive_rate_norm': 'numeric',
}

gps_cols = [convert_gs_save_string(f"{col}_gps-{x}") for x in range(1, 8) for col in target_loc_cols_template]
column_specs.update({
    gps_col: 'numeric' for gps_col in gps_cols
})
bts_cols = [convert_gs_save_string(f"{col}_bts-{x}") for x in range(1, 8) for col in target_loc_cols_template]
column_specs.update({
    bts_col: 'numeric' for bts_col in bts_cols
})
