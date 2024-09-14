import torch
import logging

from common.file_setting import find_datasets_path, make_check_points
from common.utils import set_optimizer, logging_deepfm_config

field_index = [0,
               1,
               2,
               3,
               4,
               5,
               6,
               7,
               7,
               7,
               7,
               7,
               7,
               7,
               8,
               8,
               8,
               8,
               8,
               8,
               8,
               9,
               9,
               9,
               9,
               9,
               9,
               9,
               9,
               9,
               9,
               9,
               9,
               9,
               9,
               10,
               10,
               10,
               10,
               10,
               10,
               11,
               11,
               11,
               11,
               11,
               12]

field_dict = {
    0: ["age"],

    1: ["fnlwgt"],

    2: ["education_num"],

    3: ["gender"],

    4: ["capital_gain"],

    5: ["capital_loss"],

    6: ["hours_per_week"],

    7: ["workclass_Self-emp-not-inc",
        "workclass_Federal-gov",
        "workclass_Self-emp-inc",
        "workclass_Without-pay",
        "workclass_Private",
        "workclass_Local-gov",
        "workclass_State-gov"],

    8: ["marital_status_Divorced",
        "marital_status_Married-AF-spouse",
        "marital_status_Married-civ-spouse",
        "marital_status_Married-spouse-absent",
        "marital_status_Never-married",
        "marital_status_Separated",
        "marital_status_Widowed"],

    9: ["occupation_Adm-clerical",
        "occupation_Armed-Forces",
        "occupation_Craft-repair",
        "occupation_Exec-managerial",
        "occupation_Farming-fishing",
        "occupation_Handlers-cleaners",
        "occupation_Machine-op-inspct",
        "occupation_Other-service",
        "occupation_Priv-house-serv",
        "occupation_Prof-specialty",
        "occupation_Protective-serv",
        "occupation_Sales",
        "occupation_Tech-support",
        "occupation_Transport-moving"],

    10: ["relationship_Husband",
         "relationship_Not-in-family",
         "relationship_Other-relative",
         "relationship_Own-child",
         "relationship_Unmarried",
         "relationship_Wife"],

    11: ["race_Amer-Indian-Eskimo",
         "race_Asian-Pac-Islander",
         "race_Black",
         "race_Other",
         "race_White"],

    12: ["native_country"]
}

# Operation flow sequence 1.
try:
    # setting device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
except Exception as err:
    logging.error(err)
