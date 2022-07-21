from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, precision, recall, Loss
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.handlers import EarlyStopping
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MyTrainer():
    def __init__(self, model, optimizer, loss_func, train_loader, test_loader, patience = 20):
        val_metrics = {
            'accuracy': Accuracy(output_transform=self.output_transform),
            'precision': precision.Precision(output_transform=self.output_transform),
            'recall': recall.Recall(output_transform=self.output_transform),
            'roc_auc': ROC_AUC(),
            'roc_curve': RocCurve(),
            'loss':Loss(loss_func)
        }
                    
        self.trainer = create_supervised_trainer(model, optimizer, loss_func)

        self.evaluator = create_supervised_evaluator(model, metrics = val_metrics)
        self.train_evaluator = create_supervised_evaluator(model, metrics = val_metrics)
        
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_training_results)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.log_validation_results)
        self.trainer.add_event_handler(Events.COMPLETED, self.print_roc_curve)
        if patience > 0 :
            self.handler = EarlyStopping(patience=patience, score_function=self.score_function, trainer=self.trainer)
            self.evaluator.add_event_handler(Events.COMPLETED, self.handler)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_loss = []
        self.vali_loss = []
        
    def run(self, max_epoch):
        self.trainer.run(self.train_loader, max_epochs=max_epoch)

    def score_function(self, engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss
       
    def output_transform(self, out):
        
        y_pred, y = out
        pred = torch.round(y_pred)
        return pred, y
    
    def log_training_loss(self, engine):
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration

        if i % 50 == 0:
            print(f"Epoch {e}/{n} : {i} - batch loss: {engine.state.output:.4f}")
        
    def log_training_results(self, trainer):
        self.train_evaluator.run(self.train_loader)
        metrics = self.train_evaluator.state.metrics        
        self.train_loss.append(metrics['loss'])
        
        print(f"Training Results - Epoch[{trainer.state.epoch}]\n" + 
            f"Loss: {metrics['loss']:.4f}\n" +
            f"Accuracy: {metrics['accuracy']:.4f}\n" + 
            f"Precision: {metrics['precision']:.4f}\n" +
            f"Recall: {metrics['recall']:.4f}\n" +
            f"ROC_AUC: {metrics['roc_auc']:.4f}\n")
             

    def log_validation_results(self, trainer):
        self.evaluator.run(self.test_loader)
        metrics = self.evaluator.state.metrics
        self.vali_loss.append(metrics['loss'])
        
        print(f"Validation Results - Epoch[{trainer.state.epoch}]\n" + 
            f"Loss: {metrics['loss']:.4f}\n" +
            f"Accuracy: {metrics['accuracy']:.4f}\n" +
            f"Precision: {metrics['precision']:.4f}\n" +
            f"Recall: {metrics['recall']:.4f}\n" +
            f"ROC_AUC: {metrics['roc_auc']:.4f}\n")

        
    def print_roc_curve(self, trainer):
        self.evaluator.run(self.test_loader)
        metrics = self.evaluator.state.metrics
        fpr, tpr = metrics['roc_curve'][0], metrics['roc_curve'][1]
        
        plt.plot(fpr, tpr, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend()
        plt.show()
        
    
    def print_loss_graph(self, file_name = None):
        plt.plot(self.train_loss, label='train loss')
        plt.plot(self.vali_loss, label='valid loss')
        plt.legend()
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        # plt.yscale("logit")
        plt.title("loss value graph", fontsize=14)
        
        
        if file_name is not None :
            plt.savefig(file_name + ".png")
    
        plt.show()
         


