from __future__ import annotations

OUTCOME_COL = "speaking_score"

ID_COLUMNS = [
    "dataset",
    "observation",
    "native_language",
    "native_code",
    "target_language",
    "target_code",
]

DISTANCE_FAMILIES = {
    "lexical": [
        "asjp_lexical",
        "concepts_lexical",
    ],
    "genetic": [
        "glottolog_genetic",
        "l2v_genetic",
        "urielplus_genetic",
        "modality_hyperbolic",
    ],
    "geographic": [
        "l2v_geographic",
        "urielplus_geographic",
        "modality_speaker",
    ],
    "typological_structural": [
        "grambank_typological",
        "qwals_typological",
        "l2v_featural",
        "l2v_syntactic",
        "urielplus_featural",
        "urielplus_syntactic",
        "urielplus_morphological",
        "modality_island",
    ],
    "phonological_inventory": [
        "l2v_inventory",
        "l2v_phonological",
        "phoible_phonological",
        "urielplus_inventory",
        "urielplus_phonological",
    ],
    "script": [
        "scripts_script",
        "urielplus_script",
    ],
}

LEARNER_CONFOUNDER_COLS = [
    "age_at_arrival",
    "residence_length",
    "target_education_days",
    "course_enrolment",
    "lexicon_score",
    "morphology_score",
    "new_feature_score",
    "new_sound_score",
]

COUNTRY_CONFOUNDER_COLS = [
    "hdi",
    "gdp_per_capita",
    "adult_literacy",
    "internet_use",
    "education_expenditure",
    "tertiary_enrolment",
    "female_tertiary_enrolment",
    "male_tertiary_enrolment",
    "urban_population",
]

CATEGORICAL_CONFOUNDER_COLS = [
    "sex",
]

AUXILIARY_TARGET_PREFIX = "auxiliary_target_distance_"
NATIVE_AUXILIARY_PREFIX = "native_auxiliary_distance_"