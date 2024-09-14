import torch
from torch.utils.data import DataLoader
from typing import Type
import datetime
import numpy as np
import logging
import time
from sklearn import metrics
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 초기화 파일에서 선언한 함수 불러오기.
import config
from data_loaders.census_income_eval import Census_Income
from model.DeepFM import DeepFM

from __init__ import device, find_datasets_path, logging_deepfm_config


def eval_model(model, loader, epoch):

    # 일반화 성능 검증
    with torch.no_grad():
        for data in loader:
            logging.debug(
                "\n" + "\n" + " ###################### " + \
                "\n" + " ##### start test ##### " + \
                "\n" + " ###################### " + "\n"
            )

            question, response = data
            
            model.eval()
            logging.debug("\n" + "\n" + " ############################# " + \
                          "\n" + " ######## input data ######### " + \
                          "\n" + " ############################# " + "\n" + "\n"
                          )
            # 텐서를 NumPy 배열로 변환
            numpy_array = question.numpy()

            # NumPy 배열을 리스트로 변환
            list_from_numpy = numpy_array.tolist()

            logging.debug("\n" + "\n" + " ############################# " + \
                          "\n" + " ###### question data ######## " + \
                          "\n" + " ############################# " + \
                          "\n" + str(list_from_numpy) + "\n"
                          )

            logging.debug("\n" + "\n" + " ############################# " + \
                          "\n" + " ###### response data ######## " + \
                          "\n" + " ############################# " + \
                          "\n" + str(np.array(response)) + "\n"
                          )

            predict = np.array(model(question).detach().cpu())

            logging.debug("\n" + "\n" + " #############################  " + \
                          "\n" + " ###### predict data ######### " + \
                          "\n" + " #############################  " + \
                          "\n" + str(np.array(predict)) + "\n"
                          )

            true_score = np.array(response.detach().cpu()).squeeze()
            logging.debug("\n" + "\n" + " ################################ " + \
                          "\n" + " ###### true_score data ######### " + \
                          "\n" + " ################################ " + \
                          "\n" + str(np.array(true_score)) + "\n"
                          )
            binary_predictions = np.where(predict >= 0.5, 1, 0)

            accuracy = metrics.accuracy_score(
                y_true=true_score, y_pred=binary_predictions
            )

            logging.debug(
                "\n" + "\n" + " ################## " + \
                "\n" + " ##### Result ##### " + \
                "\n" + " ################## " + \
                "\n" + " Epoch: {},   AUC: {}".format(epoch, accuracy) + "\n"
            )

    
class RunModel:

    # Operation flow sequence 3-1.
    def __init__(
            self,
            model_name=Type[str],
            dataset_name=Type[str],
            date_info=None
    ) -> None:
        """
        Initialization function to initialize parameters.

        Args:
            model_name: The model name.
            dataset_name: The dataset name.
            date_info: Datetime now.
        """
        # Operation flow sequence 3-1-1.
        self.model = None
        self.dataset_name = dataset_name
        self.date_info = date_info
        self.model_name = str(model_name)
        self.number_epochs = int(1)
        
        _ , self.ckeckpoint_path = find_datasets_path(file_name="model.ckpt")

        # Select dataset.
        # Operation flow sequence 3-1-3.
        if self.dataset_name == "Census_Income":
            self.dataset = Census_Income()

        self.dropout = float(0.5)

    # Operation flow sequence 3-2.
    def run_model(self):
        """
        Function to train the model.

        Args:
            Initialized parameters.
        """
        # Select model.
        # Operation flow sequence 3-2-1.
        
        self.model = DeepFM(
            embedding_size=config.EMBEDDING_SIZE,
            number_feature=len(self.dataset.field_index),
            number_field=len(self.dataset.field_dict),
            field_index=self.dataset.field_index,
            dropout=self.dropout
        ).to(
            device
        )
        ckeckpoint = torch.load(self.ckeckpoint_path, map_location=device)
        
        # 연결
        self.model.load_state_dict(ckeckpoint)

        logging_deepfm_config(number_epochs=self.number_epochs,
                              batch_size=int(self.dataset.length),
                              optimizer="SGD",
                              train_ratio=0.8,
                              learning_rate=0.01,
                              embedding_size=config.EMBEDDING_SIZE,
                              number_feature=len(self.dataset.field_index),
                              number_field=len(self.dataset.field_dict),                       
                              field_index=self.dataset.field_index,
                              dropout=self.dropout,
                              model=self.model)

        logging.debug(
            "\n" + "\n" + " ############################" + \
            "\n" + " ### start evaluate model ###" + \
            "\n" + " ############################" + "\n"
        )

        start = time.time()
        
        for epoch in range(1, self.number_epochs + 1):            

            test_loader = DataLoader(
                self.dataset,
                batch_size=int(self.dataset.length),
                generator=torch.Generator(device=device),
                shuffle=True
            )  

            eval_model(
                model=self.model,
                loader=test_loader,
                epoch=epoch
            )
        
        end = time.time()
        sec = (end - start)
        result_list = str(datetime.timedelta(seconds=sec)).split(".")
        # 필요하면
        # logging.debug(" total train time : %s", result_list[0])
           