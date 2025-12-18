"""
@created: 23.03.2023
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: Modus Selecter for ErgoCom
"""

from __future__ import print_function

import logging
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from network_user_proglove import Network_User

# from sacred import Experiment


class Modus_Selecter(object):
    """
    classdocs
    """

    def __init__(self, config, exp=None):
        """
        Constructor
        """

        logging.info("    Network_selecter: Constructor")
        self.config = config
        logging.info("    Network_selecter: \n{}".format(config))

        self.exp = exp
        self.network = Network_User(config, self.exp)
        self.xml_root = ET.Element("Experiment_{}".format(self.config["name_counter"]))

        return

    def save(
        self,
        acc_test,
        f1_weighted_test,
        f1_mean_test,
        ea_iter,
        type_simple="training",
        confusion_matrix=0,
        time_iter=0,
        precisions=0,
        recalls=0,
        best_itera=0,
        acc_test_seg=[0],
        f1_weighted_test_seg=[0],
        f1_mean_test_seg=[0],
    ):
        """
        Save the results of traiing and testing according to the configuration.
        As training is repeated several times, results are appended, and mean and std of all the repetitions
        are computed.

        @param acc_test: List of accuracies of val or testing
        @param f1_weighted_test: List of F1w of val or testing
        @param f1_mean_test: List of F1m of val or testing
        @param ea_iter: Iteration of evolution
        @param type_simple: Type of experiment
        @param confusion_matrix: Confusion Matrix
        @param time_iter: Time of experiment run
        @param precisions: List of class precisions
        @param recalls: List of class recalls
        @param best_itera: Best evolution iteration
        """

        child_network = ET.SubElement(self.xml_root, "network", network=str(self.config["network"]))
        child_dataset = ET.SubElement(child_network, "dataset", dataset=str(self.config["dataset"]))
        child = ET.SubElement(child_dataset, "usage_modus", usage_modus=str(self.config["usage_modus"]))
        child = ET.SubElement(
            child_dataset, "dataset_finetuning", dataset_finetuning=str(self.config["dataset_finetuning"])
        )
        child = ET.SubElement(child_dataset, "type_simple", type_simple=str(type_simple))
        child = ET.SubElement(child_dataset, "output", output=str(self.config["output"]))
        child = ET.SubElement(child_dataset, "pooling", pooling=str(self.config["pooling"]))

        child = ET.SubElement(child_dataset, "lr", lr=str(self.config["lr"]))
        child = ET.SubElement(child_dataset, "epochs", epochs=str(self.config["epochs"]))
        child = ET.SubElement(child_dataset, "distance", distance=str(self.config["distance"]))
        child = ET.SubElement(child_dataset, "reshape_input", reshape_input=str(self.config["reshape_input"]))

        child = ET.SubElement(child_dataset, "ea_iter", ea_iter=str(ea_iter))
        child = ET.SubElement(child_dataset, "freeze_options", freeze_options=str(self.config["freeze_options"]))
        child = ET.SubElement(child_dataset, "time_iter", time_iter=str(time_iter))
        child = ET.SubElement(child_dataset, "best_itera", best_itera=str(best_itera))

        for exp in range(len(acc_test)):
            child = ET.SubElement(
                child_dataset,
                "metrics",
                acc_test=str(acc_test[exp]),
                f1_weighted_test=str(f1_weighted_test[exp]),
                f1_mean_test=str(f1_mean_test[exp]),
            )
        child = ET.SubElement(
            child_dataset,
            "metrics_mean",
            acc_test_mean=str(np.mean(acc_test)),
            f1_weighted_test_mean=str(np.mean(f1_weighted_test)),
            f1_mean_test_mean=str(np.mean(f1_mean_test)),
        )
        child = ET.SubElement(
            child_dataset,
            "metrics_std",
            acc_test_std=str(np.std(acc_test)),
            f1_weighted_test_std=str(np.std(f1_weighted_test)),
            f1_mean_test_std=str(np.std(f1_mean_test)),
        )
        child = ET.SubElement(child_dataset, "confusion_matrix_last", confusion_matrix_last=str(confusion_matrix))
        if type_simple == "training":
            child = ET.SubElement(child_dataset, "precision", precision=str(precisions))
            child = ET.SubElement(child_dataset, "recall", recall=str(recalls))
        else:
            child = ET.SubElement(child_dataset, "precision_mean", precision_mean=str(np.mean(precisions, axis=0)))
            child = ET.SubElement(child_dataset, "precision_std", precision_std=str(np.std(precisions, axis=0)))
            child = ET.SubElement(child_dataset, "recall_mean", recall_mean=str(np.mean(recalls, axis=0)))
            child = ET.SubElement(child_dataset, "recall_std", recall_std=str(np.std(recalls, axis=0)))

        for exp in range(len(acc_test_seg)):
            child = ET.SubElement(
                child_dataset,
                "metrics_seg",
                acc_test_seg=str(acc_test_seg[exp]),
                f1_weighted_test_seg=str(f1_weighted_test_seg[exp]),
                f1_mean_test_seg=str(f1_mean_test_seg[exp]),
            )
        child = ET.SubElement(
            child_dataset,
            "metrics_seg_mean",
            acc_test_seg_mean=str(np.mean(acc_test_seg)),
            f1_weighted_test_seg_mean=str(np.mean(f1_weighted_test_seg)),
            f1_mean_test_seg_mean=str(np.mean(f1_mean_test_seg)),
        )
        child = ET.SubElement(
            child_dataset,
            "metrics_seg_std",
            acc_test_seg_std=str(np.std(acc_test_seg)),
            f1_weighted_test_seg_std=str(np.std(f1_weighted_test_seg)),
            f1_mean_test_seg_std=str(np.std(f1_mean_test_seg)),
        )

        return

    def train(self, itera=1, testing=False):
        """
        Train method. Train network for a certain number of repetitions
        computing the val performance, Testing using test(), saving the performances

        @param itera: training iteration, as training is repeated X number of times
        @param testing: Enabling testing after training
        """

        global f1_weighted_test_ac
        logging.info("    Network_selecter: Train")

        acc_train_ac = []
        f1_weighted_train_ac = []
        f1_mean_train_ac = []
        precisions_test = []
        recalls_test = []
        acc_train_seg_ac = []
        f1_weighted_train_seg_ac = []
        f1_mean_train_seg_ac = []

        if testing:
            acc_test_ac = []
            f1_weighted_test_ac = []
            f1_mean_test_ac = []
            acc_test_seg_ac = []
            f1_weighted_test_seg_ac = []
            f1_mean_test_seg_ac = []

        # There will be only one iteration
        # As there is not evolution
        for iter_evl in range(itera):
            start_time_train = time.time()
            logging.info("    Network_selecter:    Train iter 0")
            # Training the network and obtaining the validation results
            results_train, confusion_matrix_train, best_itera = self.network.evolution_evaluation(ea_iter=iter_evl)

            # Appending results for later saving in results file
            acc_train_ac.append(results_train["classification"]["acc"])
            f1_weighted_train_ac.append(results_train["classification"]["f1_weighted"])
            f1_mean_train_ac.append(results_train["classification"]["f1_mean"])
            acc_train_seg_ac.append(results_train["segmentation"]["acc"])
            f1_weighted_train_seg_ac.append(results_train["segmentation"]["f1_weighted"])
            f1_mean_train_seg_ac.append(results_train["segmentation"]["f1_mean"])

            time_train = time.time() - start_time_train

            logging.info(
                "    Network_selecter:    Train: elapsed time {} acc {}, "
                "f1_weighted {}, f1_mean {}".format(
                    time_train,
                    results_train["classification"]["acc"],
                    results_train["classification"]["f1_weighted"],
                    results_train["classification"]["f1_mean"],
                )
            )
            # Saving the results
            if self.config["tensorboard_bool"]:
                self.exp.add_scalar("Acc_Val", scalar_value=results_train["classification"]["acc"])
                self.exp.add_scalar("F1w_Val", scalar_value=results_train["classification"]["f1_weighted"])
                self.exp.add_scalar("F1m_Val", scalar_value=results_train["classification"]["f1_mean"])
                self.exp.add_scalar("Acc_Seg_Val", scalar_value=results_train["segmentation"]["acc"])
                self.exp.add_scalar("F1w_Seg_Val", scalar_value=results_train["segmentation"]["f1_weighted"])
                self.exp.add_scalar("F1m_Seg_Val", scalar_value=results_train["segmentation"]["f1_mean"])
                self.exp.add_scalar("best_itera", scalar_value=best_itera)

            self.save(
                acc_train_ac,
                f1_weighted_train_ac,
                f1_mean_train_ac,
                ea_iter=iter_evl,
                time_iter=time_train,
                precisions=results_train["classification"]["precision"],
                recalls=results_train["classification"]["recall"],
                best_itera=best_itera,
                acc_test_seg=acc_train_seg_ac,
                f1_weighted_test_seg=f1_weighted_train_seg_ac,
                f1_mean_test_seg=f1_mean_train_seg_ac,
            )

            # Testing the network
            if testing:
                start_time_test = time.time()
                results_test, confusion_matrix_test = self.test(testing=True)
                acc_test_ac.append(results_test["classification"]["acc"])
                f1_weighted_test_ac.append(results_test["classification"]["f1_weighted"])
                f1_mean_test_ac.append(results_test["classification"]["f1_mean"])
                acc_test_seg_ac.append(results_test["segmentation"]["acc"])
                f1_weighted_test_seg_ac.append(results_test["segmentation"]["f1_weighted"])
                f1_mean_test_seg_ac.append(results_test["segmentation"]["f1_mean"])
                precisions_test.append(results_test["classification"]["precision"].numpy())
                recalls_test.append(results_test["classification"]["recall"].numpy())

                time_test = time.time() - start_time_test

                if self.config["tensorboard_bool"]:
                    self.exp.add_scalar("Acc_Test", scalar_value=results_test["classification"]["acc"])
                    self.exp.add_scalar("F1w_Test", scalar_value=results_test["classification"]["f1_weighted"])
                    self.exp.add_scalar("F1m_Test", scalar_value=results_test["classification"]["f1_mean"])
                    self.exp.add_scalar("Acc_Seg_Test", scalar_value=results_test["segmentation"]["acc"])
                    self.exp.add_scalar("F1w_Seg_Test", scalar_value=results_test["segmentation"]["f1_weighted"])
                    self.exp.add_scalar("F1m_Seg_Test", scalar_value=results_test["segmentation"]["f1_mean"])
                    self.save(
                        acc_test_ac,
                        f1_weighted_test_ac,
                        f1_mean_test_ac,
                        ea_iter=iter_evl,
                        type_simple="testing",
                        confusion_matrix=confusion_matrix_test,
                        time_iter=time_test,
                        precisions=np.array(precisions_test),
                        recalls=np.array(recalls_test),
                        acc_test_seg=acc_test_seg_ac,
                        f1_weighted_test_seg=f1_weighted_test_seg_ac,
                        f1_mean_test_seg=f1_mean_test_seg_ac,
                    )

        if testing:
            self.save(
                acc_test_ac,
                f1_weighted_test_ac,
                f1_mean_test_ac,
                ea_iter=iter_evl,
                type_simple="testing",
                confusion_matrix=confusion_matrix_test,
                time_iter=time_test,
                precisions=np.array(precisions_test),
                recalls=np.array(recalls_test),
                acc_test_seg=acc_test_seg_ac,
                f1_weighted_test_seg=f1_weighted_test_seg_ac,
                f1_mean_test_seg=f1_mean_test_seg_ac,
            )

        if self.config["usage_modus"] == "train":
            logging.info("    Network_selecter:    Train:    eliminating network file")
            os.remove(self.config["folder_exp"] + "network.pt")

        return

    def test(self, testing=False):
        """
        Test method. Testing the network , saving the performances

        @param itera: training iteration, as training is repeated X number of times
        @param testing: Enabling testing after training
        @return results_test: dict with the results of the testing
        @return confusion_matrix_test: confusion matrix of the text
        """

        start_time_test = time.time()
        precisions_test = []
        recalls_test = []

        # Testing the network in folder (according to the conf)
        results_test, confusion_matrix_test, _ = self.network.evolution_evaluation(ea_iter=0, testing=testing)

        elapsed_time_test = time.time() - start_time_test

        # Appending results for later saving in results file
        precisions_test.append(results_test["classification"]["precision"].numpy())
        recalls_test.append(results_test["classification"]["recall"].numpy())

        logging.info(
            "    Network_selecter:    Train: elapsed time {} acc {}, "
            "f1_weighted {}, f1_mean {}".format(
                elapsed_time_test,
                results_test["classification"]["acc"],
                results_test["classification"]["f1_weighted"],
                results_test["classification"]["f1_mean"],
            )
        )

        # Saving the results
        if not testing:
            if self.config["tensorboard_bool"]:
                self.exp.add_scalar("Acc_Test", scalar_value=results_test["classification"]["acc"])
                self.exp.add_scalar("F1w_Test", scalar_value=results_test["classification"]["f1_weighted"])
                self.exp.add_scalar("F1m_Test", scalar_value=results_test["classification"]["f1_mean"])
                self.exp.add_scalar("Acc_Seg_Test", scalar_value=results_test["segmentation"]["acc"])
                self.exp.add_scalar("F1w_Seg_Test", scalar_value=results_test["segmentation"]["f1_weighted"])
                self.exp.add_scalar("F1m_Seg_Test", scalar_value=results_test["segmentation"]["f1_mean"])
                # self.exp.add_scalar("Precision_Test", value=str(results_test['precision'].numpy()))
                # self.exp.add_scalar("Recall_Test", value=str(results_test['recall'].numpy()))
                # self.exp.add_scalar("Confusion_Matrix_Test", value=str(confusion_matrix_test))

            self.save(
                [results_test["classification"]["acc"]],
                [results_test["classification"]["f1_weighted"]],
                [results_test["classification"]["f1_mean"]],
                ea_iter=0,
                type_simple="testing",
                confusion_matrix=confusion_matrix_test,
                time_iter=elapsed_time_test,
                precisions=np.array(precisions_test),
                recalls=np.array(recalls_test),
                acc_test_seg=[results_test["segmentation"]["acc"]],
                f1_weighted_test_seg=[results_test["segmentation"]["f1_weighted"]],
                f1_mean_test_seg=[results_test["segmentation"]["f1_mean"]],
            )
            return

        return results_test, confusion_matrix_test

    def net_modus(self):
        """
        Setting the training, validation, and final training.
        """
        logging.info("    Network_selecter: Net modus: {}".format(self.config["usage_modus"]))
        if self.config["usage_modus"] == "train":
            self.train(itera=1, testing=True)
        elif self.config["usage_modus"] == "test":
            self.test()
        elif self.config["usage_modus"] == "train_final":
            self.train(itera=1, testing=True)
        elif self.config["usage_modus"] == "fine_tuning":
            self.train(itera=1, testing=True)

        xml_file_path = self.config["folder_exp"] + self.config["file_suffix"]
        xmlstr = minidom.parseString(ET.tostring(self.xml_root)).toprettyxml(indent="   ")
        with open(xml_file_path, "a") as f:
            f.write(xmlstr)

        print(xmlstr)

        return
