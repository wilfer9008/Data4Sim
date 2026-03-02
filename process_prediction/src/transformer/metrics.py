"""
@created: 16.03.2021
@author: Fernando Moya Rueda

@copyright: Motion Miners GmbH, Emil-Figge Str. 76, 44227 Dortmund, 2022

@brief: Metrics using GPU
"""


import logging

import numpy as np
import torch


class Metrics(object):
    """
    classdocs
    """

    def __init__(self, config, dev, attributes, testing=False):
        """
        Constructor.
        """

        logging.info("            Metrics: Constructor")
        self.config = config
        self.device = dev
        self.mode = "Process"
        self.testing = testing
        # Here, you need to extract the attributes from the network.pt
        # self,attr= network["att_rep"]
        self.attr = attributes
        if self.config["distance"] == "euclidean":
            for attr_idx in range(self.attr.shape[0]):
                self.attr[attr_idx, 1:] = self.attr[attr_idx, 1:] / np.linalg.norm(self.attr[attr_idx, 1:])

        self.atts = torch.from_numpy(self.attr).type(dtype=torch.FloatTensor)

        if torch.cuda.is_available():
            self.atts = self.atts.type(dtype=torch.cuda.FloatTensor)
        else:
            self.atts = self.atts.type(dtype=torch.FloatTensor)

        self.results = {
            "Process": {
                "acc": 0,
                "f1_weighted": 0,
                "f1_mean": 0,
                "predicted_classes": 0,
                "precision": 0,
                "recall": 0,
            },
            "Activity": {
                "acc": 0,
                "f1_weighted": 0,
                "f1_mean": 0,
                "predicted_classes": 0,
                "precision": 0,
                "recall": 0,
            },
            "Activity_windows": {
                "acc": 0,
                "f1_weighted": 0,
                "f1_mean": 0,
                "predicted_classes": 0,
                "precision": 0,
                "recall": 0,
            },
            "Attributes": {
                "acc": 0,
                "f1_weighted": 0,
                "f1_mean": 0,
                "predicted_classes": 0,
                "precision": 0,
                "recall": 0,
            },
            "Attributes_windows": {
                "acc": 0,
                "f1_weighted": 0,
                "f1_mean": 0,
                "predicted_classes": 0,
                "precision": 0,
                "recall": 0,
            },
            "Statistics_activities": {
                "acc": 0,
                "f1_weighted": 0,
                "f1_mean": 0,
                "predicted_classes": 0,
                "precision": 0,
                "recall": 0,
            },
            "classification": {
                "acc": 0,
                "f1_weighted": 0,
                "f1_mean": 0,
                "predicted_classes": 0,
                "precision": 0,
                "recall": 0,
            },
            "segmentation": {
                "acc": 0,
                "f1_weighted": 0,
                "f1_mean": 0,
                "predicted_classes": 0,
                "precision": 0,
                "recall": 0,
            },
        }

        return

    ##################################################
    ###########  Precision and Recall ################
    ##################################################
    def get_precision_recall(self, targets, predictions):
        """
        Compute the precision and recall for all the activity classes

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return precision: torch array with precision of each class
        @return recall: torch array with recall of each class
        """

        if self.mode == "Process":
            nc = 8
        else:
            nc = self.config["num_classes"]

        precision = torch.zeros((nc))
        recall = torch.zeros((nc))

        x = torch.ones(predictions.size())
        y = torch.zeros(predictions.size())

        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)

        for c in range(nc):
            selected_elements = torch.where(predictions == c, x, y)
            non_selected_elements = torch.where(predictions == c, y, x)

            target_elements = torch.where(targets == c, x, y)
            non_target_elements = torch.where(targets == c, y, x)

            true_positives = torch.sum(target_elements * selected_elements)
            false_positives = torch.sum(non_target_elements * selected_elements)

            false_negatives = torch.sum(target_elements * non_selected_elements)
            #if self.testing:
            #    true_negatives = torch.sum(non_target_elements * non_selected_elements)

            #    acc_per_class = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives +
            #                                                         false_negatives)

            #    logging.info("            Metric:    \nAccuracy per class {}: \n{}\n".format(c, acc_per_class))

            try:
                precision[c] = true_positives.item() / float((true_positives + false_positives).item())
                recall[c] = true_positives.item() / float((true_positives + false_negatives).item())

            except:
                # logging.error('        Network_User:    Train:    In Class {} true_positives {} false_positives {} false_negatives {}'.format(c, true_positives.item(),
                #                                                                                                                              false_positives.item(),
                #                                                                                                                              false_negatives.item()))
                continue

        return precision, recall

    #############################################################
    ###########  Precision and Recall Attributes ################
    #############################################################

    def get_precision_recall_attrs(self, targets, predictions):
        precision = torch.zeros((self.config["num_attributes"]))
        recall = torch.zeros((self.config["num_attributes"]))

        x = torch.ones(predictions.size()[0])
        y = torch.zeros(predictions.size()[0])

        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)

        for c in range(self.config["num_attributes"]):
            selected_elements = torch.where(predictions[:, c] == 1.0, x, y)
            non_selected_elements = torch.where(predictions[:, c] == 1.0, y, x)

            target_elements = torch.where(targets[:, c] == 1.0, x, y)
            non_target_elements = torch.where(targets[:, c] == 1.0, y, x)

            true_positives = torch.sum(target_elements * selected_elements)
            false_positives = torch.sum(non_target_elements * selected_elements)

            false_negatives = torch.sum(target_elements * non_selected_elements)

            try:
                precision[c] = true_positives.item() / float((true_positives + false_positives).item())
                recall[c] = true_positives.item() / float((true_positives + false_negatives).item())

            except:
                # logging.error('        Network_User:    Train:    In Class {} true_positives {} false_positives {} false_negatives {}'.format(c, true_positives.item(),
                #                                                                                                                              false_positives.item(),
                #                                                                                                                              false_negatives.item()))
                continue

        return precision, recall

    ##################################################
    #################  F1 metric  ####################
    ##################################################

    def f1_metric(self, targets, preds, attrs_prediction_on=False):
        """
        Compute the f1 metrics

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return F1_weighted: F1 weighted
        @return F1_mean: F1 mean
        """

        if self.mode == "Process":
            nc = 8
        else:
            nc = self.config["num_classes"]

        if self.mode in ['Process', 'Activity', 'Activity_windows', 'Attributes', 'Attributes_windows']:
            # Predictions
            if self.config["output"] == "softmax":
                if attrs_prediction_on:
                    predictions = self.atts[torch.argmin(preds, dim=1), 0]
                else:
                    predictions = torch.argmax(preds, dim=1)
            elif self.config["output"] == "attribute":
                # predictions = torch.argmin(preds, dim=1)
                predictions = self.atts[torch.argmin(preds, dim=1), 0]
            elif self.config["output"] == "identity":
                predictions = torch.argmax(preds, dim=1)

            if self.config["output"] == "softmax":
                if attrs_prediction_on:
                    precision, recall = self.get_precision_recall(targets[:, 0], predictions)
                else:
                    precision, recall = self.get_precision_recall(targets, predictions)
            elif self.config["output"] == "attribute":
                precision, recall = self.get_precision_recall(targets[:, 0], predictions)
            elif self.config["output"] == "identity":
                precision, recall = self.get_precision_recall(targets, predictions)
        elif self.mode == "segmentation":
            predictions = preds
            precision, recall = self.get_precision_recall(targets, predictions)

        proportions = torch.zeros(nc)

        if self.config["output"] == "softmax":
            if attrs_prediction_on:
                for c in range(nc):
                    if self.mode in ['Process', 'Activity', 'Activity_windows', 'Attributes', 'Attributes_windows']:
                        proportions[c] = torch.sum(targets[:, 0] == c).item() / float(targets[:, 0].size()[0])
                    elif self.mode == "segmentation":
                        proportions[c] = torch.sum(targets == c).item() / float(targets.size()[0])
            else:
                for c in range(nc):
                    proportions[c] = torch.sum(targets == c).item() / float(targets.size()[0])
        elif self.config["output"] == "attribute":
            for c in range(nc):
                if self.mode in ['Process', 'Activity', 'Activity_windows', 'Attributes', 'Attributes_windows']:
                    proportions[c] = torch.sum(targets[:, 0] == c).item() / float(targets[:, 0].size()[0])
                elif self.mode == "segmentation":
                    proportions[c] = torch.sum(targets == c).item() / float(targets.size()[0])
        elif self.config["output"] == "identity":
            for c in range(nc):
                proportions[c] = torch.sum(targets == c).item() / float(targets.size()[0])

        logging.info("            Metric:    \nPrecision: \n{}\nRecall\n{}".format(precision, recall))
        logging.info("            Metric:    \nProportions: \n{}\n".format(proportions))

        self.results[self.mode]["precision"] = precision
        self.results[self.mode]["recall"] = recall

        self.results["classification"]["precision"] = precision
        self.results["classification"]["recall"] = recall

        multi_pre_rec = precision * recall
        sum_pre_rec = precision + recall

        multi_pre_rec[torch.isnan(multi_pre_rec)] = 0
        sum_pre_rec[torch.isnan(sum_pre_rec)] = 0

        # F1 weighted
        weighted_f1 = proportions * (multi_pre_rec / sum_pre_rec)
        weighted_f1[torch.isnan(weighted_f1)] = 0

        F1_weighted = torch.sum(weighted_f1) * 2

        # F1 mean
        f1 = multi_pre_rec / sum_pre_rec
        f1[torch.isnan(f1)] = 0

        F1_mean = torch.sum(f1) * 2 / nc

        return F1_weighted.item(), F1_mean.item()

    ##################################################
    #################  Accuracy  ####################
    ##################################################

    def acc_metric(self, targets, predictions, attrs_prediction_on=False):
        """
        Compute the Accuracy

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return acc: Accuracy
        """

        # Accuracy
        if self.mode in ['Process', 'Activity', 'Activity_windows', 'Attributes', 'Attributes_windows']:
            if self.config["output"] == "softmax":
                if attrs_prediction_on:
                    # Classes from the distances to the attribute representation
                    # with this torch.argmin(predictions, dim=1), one computes the argument where distance is min
                    # with this self.atts[torch.argmin(predictions, dim=1), 0]
                    #  one computes the class that correspond to the argument with min distance
                    # self.atts.size() = [# of windows, classes and 19 attributes] = [# of windows, 20], [#, 20]

                    predicted_classes = self.atts[torch.argmin(predictions, dim=1), 0]
                    logging.info("            Metric:    Acc:    Target     class {}".format(targets[0, 0]))
                    logging.info("            Metric:    Acc:    Prediction class {}".format(predicted_classes[0]))
                    acc = torch.sum(targets[:, 0] == predicted_classes.type(dtype=torch.cuda.FloatTensor))
                else:
                    if torch.cuda.is_available():
                        predicted_classes = torch.argmax(predictions, dim=1).type(dtype=torch.cuda.FloatTensor)
                    else:
                        predicted_classes = torch.argmax(predictions, dim=1).type(dtype=torch.FloatTensor)

                    acc = torch.sum(targets == predicted_classes)
            elif self.config["output"] == "attribute":
                # Classes from the distances to the attribute representation
                # with this torch.argmin(predictions, dim=1), one computes the argument where distance is min
                # with this self.atts[torch.argmin(predictions, dim=1), 0]
                #  one computes the class that correspond to the argument with min distance
                # self.atts.size() = [# of windows, classes and 19 attributes] = [# of windows, 20], [#, 20]

                predicted_classes = self.atts[torch.argmin(predictions, dim=1), 0]
                logging.info("            Metric:    Acc:    Target     class {}".format(targets[0, 0]))
                logging.info("            Metric:    Acc:    Prediction class {}".format(predicted_classes[0]))
                acc = torch.sum(targets[:, 0] == predicted_classes.type(dtype=torch.cuda.FloatTensor))
            elif self.config["output"] == "identity":
                predicted_classes = torch.argmax(predictions, dim=1).type(dtype=torch.cuda.FloatTensor)
                acc = torch.sum(targets == predicted_classes)
        elif self.mode == "segmentation":
            predicted_classes = predictions
            acc = torch.sum(targets == predicted_classes)
        acc = acc.item() / float(targets.size()[0])

        # returning accuracy and predicted classes
        return acc, predicted_classes

    ##################################################
    ################  Acc attr  #####################
    ##################################################

    def metric_attr(self, targets, predictions):
        """
        Compute the Accuracy per attribute or attribute vector

        @param targets: torch array with targets
        @param predictions: torch array with predictions
        @return acc_vc: Accuracy per attribute vector
        @return acc_atr: Accuracy per attribute
        """

        # logging.info('        Network_User:    Metrics')

        # preds = targets * torch.round(predictions)
        # logging.info('            Metric:    size preds vector: \n{}'.format(preds[0,:]))
        # Accuracy per vector
        # acc_vc = torch.sum(preds, dim=1, dtype=torch.cuda.FloatTensor)
        # acc_vc = torch.mean(acc_vc / float(targets.size()[1])).item()
        # logging.info('            Metric:    Acc attr vector: {}'.format(acc_vc))

        # Accuracy per attr
        acc_attrs = np.zeros(self.config["num_attributes"])
        for attr_idx in range(self.config["num_attributes"]):
            # acc_attrs[attr_idx] = torch.sum(torch.round(targets)[:, attr_idx] * torch.round(predictions)[:, attr_idx])
            acc_attrs[attr_idx] = torch.sum(torch.round(targets)[:, attr_idx] == torch.round(predictions)[:, attr_idx])
            acc_attrs[attr_idx] = acc_attrs[attr_idx] / float(targets.size()[0])

        # acc_atr = torch.sum(preds, dim=0)
        # acc_atr = acc_atr / float(targets.size()[0])
        logging.info("            Metric:    Acc attr: \n{}".format(acc_attrs))

        #logging.info("            Metric:    Acc attr: \n{}".format(acc_attrs[[24, 26, 27, 28, 29]]))

        precision_attr, recall_attr = self.get_precision_recall_attrs(targets, torch.round(predictions))
        logging.info("            Metric:    Precision attr: \n{}".format(precision_attr))
        logging.info("            Metric:    Recall attr: \n{}".format(recall_attr))
        return

    ##################################################
    ###################  metric  ######################
    ##################################################
    def mean_std_predictions_per_class(self, targets, predictions):

        logging.info("              Metric:    Prediction size {}: \n".format(predictions.size()))
        logging.info("              Metric:    Target size {}: \n".format(targets.size()))

        if self.mode == "Process":
            nc = 8
        else:
            nc = self.config["num_classes"]

        mean_class_prediction = torch.zeros((nc))
        std_class_prediction = torch.zeros((nc))
        max_class_prediction = torch.zeros((nc))
        min_class_prediction = torch.zeros((nc))

        x = torch.ones(predictions.size(0))
        y = torch.zeros(predictions.size(0))

        x = x.to(self.device, dtype=torch.long)
        y = y.to(self.device, dtype=torch.long)

        for c in range(nc):
            try:
                target_elements = torch.where(targets == c, x, y)
                mean_class_prediction[c] = torch.mean(predictions[target_elements, c])
                std_class_prediction[c] = torch.std(predictions[target_elements, c])
                max_class_prediction[c] = torch.max(predictions[target_elements, c])
                min_class_prediction[c] = torch.mean(predictions[target_elements, c])
            except:
                logging.error('        Network_User:    Train:    In Class {} target_elements.size() {}'.format(c, target_elements.size()))
                continue

        logging.info("              Metric:    \nMean Prediction per class {}\n".format(torch.round(mean_class_prediction, decimals=5)))
        logging.info("              Metric:    \nStd Prediction per class {}\n".format(torch.round(std_class_prediction, decimals=5)))
        logging.info("              Metric:    \nMax Prediction per class {}\n".format(torch.round(max_class_prediction, decimals=5)))
        logging.info("              Metric:    \nMin Prediction per class {}\n".format(torch.round(min_class_prediction, decimals=5)))

        return

    ##################################################
    ###################  metric  ######################
    ##################################################
    def efficient_distance(self, predictions):
        """
        Compute Euclidean distance from predictions (output of sigmoid) to attribute representation

        @param predictions: torch array with predictions (output from sigmoid)
        @return distances: Euclidean Distance to each of the vectors in the attribute representation
        """
        if self.config["distance"] == "euclidean":
            dist_funct = torch.nn.PairwiseDistance()

            # Normalize the predictions of the network
            for pred_idx in range(predictions.size()[0]):
                predictions[pred_idx, :] = predictions[pred_idx, :] / torch.norm(predictions[pred_idx, :])

            predictions = predictions.repeat(self.attr.shape[0], 1, 1)
            predictions = predictions.permute(1, 0, 2)

            # compute the distance among the predictions of the network
            # and the the attribute representation
            distances = dist_funct(predictions[0], self.atts[:, 1:])
            distances = distances.view(1, -1)
            for i in range(1, predictions.shape[0]):
                dist = dist_funct(predictions[i], self.atts[:, 1:])
                distances = torch.cat((distances, dist.view(1, -1)), dim=0)

        elif self.config["distance"] == "BCELoss":
            dist_funct = torch.nn.BCELoss(reduce=False, reduction="sum")

            attrs_repeat = np.reshape(self.attr, newshape=[1, self.attr.shape[0], self.attr.shape[1]])  # [1, 302,19]
            attrs_repeat = np.repeat(attrs_repeat, predictions.shape[0], axis=0)  # [batches, 302,19] = #[200, 302,19]
            attrs_repeat = torch.from_numpy(attrs_repeat[:, :, 1:])
            attrs_repeat = attrs_repeat.to(self.device, dtype=torch.float)

            predictions = predictions.repeat(self.attr.shape[0], 1, 1)  ##[200, 19] = #[302, 200,19]
            predictions = predictions.permute(1, 0, 2)  ##[200, 302,19]

            # compute the distance among the predictions of the network
            # and the the attribute representation
            if self.config["aggregate"] in ["FCN", "LSTM"]:
                distances = dist_funct(predictions[0], attrs_repeat[0])
                distances = torch.reshape(distances, shape=(1, distances.shape[0], distances.shape[1]))
                for i in range(1, predictions.shape[0]):
                    dist = dist_funct(predictions[i], attrs_repeat[i])
                    dist = torch.reshape(dist, shape=(1, dist.shape[0], dist.shape[1]))
                    distances = torch.cat((distances, dist), dim=0)
                distances = distances.sum(axis=2)
            else:
                distances = dist_funct(predictions, attrs_repeat)  # predictions [200, 302,19] vs #attr rep[200, 302,19]
                # distances [200, 302, 19]
                #### one - distances /100
                distances = distances.sum(axis=2)  # [200, 302]

        # return the distances
        return distances


    ##################################################
    ###################  metric  ######################
    ##################################################
    def return_results(self):

        return self.results
    ##################################################
    ###################  metric  ######################
    ##################################################

    def metric(self, targets, predictions, attrs_prediction_on=False, classification_target="Process"):
        self.mode = classification_target
        # logging.info('        Network_User:    Metrics')
        #if attrs_prediction_on==False:
        #    self.mean_std_predictions_per_class(targets, predictions)
        if self.config["output"] == "attribute" and self.mode == "classification":
            logging.info("\n")
            logging.info(
                "            Metric:    metric:    target example \n{}\n{}".format(targets[0, 1:], predictions[0])
            )
            logging.info("            Metric:    type targets vector: {}".format(targets.type()))
            if self.config["aggregate"] in ["FCN", "LSTM"]:
                self.metric_attr(targets, predictions)
            if self.config["aggregate"] == "FC":
                self.metric_attr(targets[:, 1:], predictions)
            predictions = self.efficient_distance(predictions)
        else:
            if attrs_prediction_on:
                logging.info("\n")
                logging.info(
                    "            Metric:    metric:    target example \n{}\n{}".format(targets[0, 1:], predictions[0])
                )
                logging.info("            Metric:    type targets vector: {}".format(targets.type()))
                if self.config["aggregate"] in ["FCN", "LSTM"]:
                    self.metric_attr(targets, predictions)
                if self.config["aggregate"] == "FC":
                    print("targets[:, 1:]: ", targets.size())
                    print("predictions: ", predictions.size())
                    self.metric_attr(targets, predictions)
                predictions = self.efficient_distance(predictions)
                print("predictions distance: ", predictions.size())

        # Accuracy
        targets = targets.type(dtype=torch.FloatTensor)
        targets = targets.to(self.device)
        # if softmax = argmax(predictions)
        # if sigmoid = argmin(predictions) why? because predcitions are in this case distances
        #
        # with self.acc_metric, one computes the classes either from softmax os attributes
        acc, predicted_classes = self.acc_metric(targets, predictions, attrs_prediction_on=attrs_prediction_on)

        # F1 metrics
        f1_weighted, f1_mean = self.f1_metric(targets, predictions, attrs_prediction_on=attrs_prediction_on)

        self.results[self.mode]["acc"] = acc
        self.results[self.mode]["f1_weighted"] = f1_weighted
        self.results[self.mode]["f1_mean"] = f1_mean
        self.results[self.mode]["predicted_classes"] = predicted_classes

        self.results["classification"]["acc"] = self.results[self.config['classification_target']]["acc"]
        self.results["classification"]["f1_weighted"] = self.results[self.config['classification_target']]["f1_weighted"]
        self.results["classification"]["f1_mean"] = self.results[self.config['classification_target']]["f1_mean"]
        self.results["classification"]["predicted_classes"] = self.results[self.config['classification_target']]["predicted_classes"]

        return
