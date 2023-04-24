""" https://github.com/sebp/survival-cnn-estimator """
from typing import Sequence, Optional
import tensorflow as tf
import numpy as np
from sksurv.linear_model.coxph import BreslowEstimator
from c_index import CIndex
from loss import CoxPHLoss
import matplotlib.pyplot as plt

# Tensorflow 2 has a config option to run functions "eagerly" which will enable getting 
# Tensor values via .numpy() method. Without this configuration we get the arribute error
tf.config.run_functions_eagerly(True)


class DeepSurvival:

    def __init__(self, 
                model:tf.keras.Model,
                learning_rate:int, 
                epochs:int
        ):
        self.epochs = epochs
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = CoxPHLoss()

        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.train_cindex_metric = CIndex()
        self.test_cindex_metric = CIndex()
        
        self.batch_size = None
        self._baseline_model = BreslowEstimator()
    
    @tf.function
    def train_step(self, X, event, time):
        """
        Parameters:
        ----------
        X:
        y: cotains te event & time
        event:
        time:
        event : Boolean
        #     The first element holds a binary vector where 1 indicates an event and 0 censoring.
        # y_riskset: Boolean
        #     A boolean matrix where the `i`-th row denotes the risk set of the `i`-th instance, i.e. 
        #     the indices `j` for which the observer time `y_j >= y_i`.
        """

        y = event, time
        with tf.GradientTape() as tape:
            pred = self.model(X, training=True)
            loss = self.loss_fn(y, pred)

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        # Update training metrics
        self.train_cindex_metric.update_state(y, pred)
        self.train_loss_metric.update_state(loss)
        return loss, pred

    @staticmethod
    def make_tf_dataset(dataset, batch_size:int, seed:int):
        """
        Making a tensorflow dataset from data
        A tuple of dataframe and series
        dataset: tuple(pd.DataFrame, pd.Series, pd.Series)
        """
        X, event, time = dataset
        dataset_size = len(dataset)
        return (
            tf.data.Dataset
            .from_tensor_slices((X, event, time))
            .shuffle(dataset_size, seed=seed)
            .batch(batch_size,  drop_remainder=False)    
        ), (serie.values for serie in dataset)

    def train_evaluate(self, train_dataset, val_dataset, batch_size:int, seed:int=42):
        self.batch_size = batch_size
        train_dataset, (X_train, event_train, time_train) = self.make_tf_dataset(train_dataset, batch_size, seed)
        val_dataset, _ = self.make_tf_dataset(val_dataset, batch_size, seed)

        for epoch in range(1, self.epochs + 1):
            # Iterate over the batches of the dataset.
            for X, event, time in train_dataset:
                loss, pred = self.train_step(X, event, time)

            # Display training metrics at the end of each epoch.
            print(f"\nEpoch: {epoch}")
            train_loss = self.train_loss_metric.result()
            train_metrics = self.train_cindex_metric.result()
            print(f"Training Loos: {train_loss.numpy():.4f}")
            print(f"Training Accuracy[C-index]: {train_metrics['cindex'].numpy():.4f}")
        
            # Reset training metrics at the end of each epoch
            self.train_cindex_metric.reset_states()

            # Run a validation loop at the end of each epoch
            for X, event, time in val_dataset:
                self._evaluate(X, event, time)
        
        # After training stack all data instances with their features and fit Breslow estimator 
        preds = self.model(X_train, training=False)
        preds = tf.squeeze(preds).numpy()
        # fit args should be 1D arrays
        self._baseline_model.fit(preds, event_train, time_train)
        return self

    @tf.function
    def _evaluate(self, X, event, time):
        y = event, time
        test_pred = self.model(X, training=False)
        self.test_cindex_metric.update_state(y, test_pred)
        test_metrics = self.test_cindex_metric.result()
        print(f"Validation Accuracy[C-index]: {test_metrics['cindex'].numpy():.4f}")
        self.test_cindex_metric.reset_states()

    def predict(self, X):
        """
        Predict risk scores.
        Parameters
        ----------
        X : DataFrame array-like, shape = (n_samples, n_features)
            Data matrix with a size less than batch size.
        Returns
        -------
        risk_score : array, shape = (n_samples,)
            Predicted risk scores.
        """
        data_batch = (
            tf.data.Dataset
            .from_tensor_slices(X.values)
            .batch(self.batch_size)
        )
        for X in data_batch:
            return self.model(X, training=False)

    def predict_survival_function(self, X):
        """
        Predict survival function.
        The survival function for an individual
        with feature vector :math:`x` is defined as
        .. math::
            S(t \\mid x) = S_0(t)^{\\exp(x^\\top \\beta)} ,
        where :math:`S_0(t)` is the baseline survival function,
        estimated by Breslow's estimator.
        """

        return self._baseline_model.get_survival_function(self.predict(X))
    
    def plot_survival_function(self, X, figsize:tuple=(8,6), *args, **kwargs):
        plt.figure(figsize=figsize)
        survival_funcs = self.predict_survival_function(X)
        for survival_func in survival_funcs:
            plt.step(survival_func.x, survival_func.y, where="post",  *args, **kwargs)
        plt.ylim(0, 1)
        plt.ylabel("Probability of survival $P(T > t)$")
        plt.xlabel("Time $t$")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    pass