import math 
import numpy as np
class LogisticRegressor:
    def __init__(self):
        """
        Initializes the Logistic Regressor model.

        Attributes:
        - weights (np.ndarray): A placeholder for the weights of the model.
                                These will be initialized in the training phase.
        - bias (float): A placeholder for the bias of the model.
                        This will also be initialized in the training phase.
        """
        self.weights = None
        self.bias = None


    def fit(
        self,
        X,
        y,
        learning_rate=0.01,
        num_iterations=1000,
        penalty=None,
        l1_ratio=0.5,
        C=1.0,
        verbose=False,
        print_every=100,
    ):
        """
        Fits the logistic regression model to the data using gradient descent.

        This method initializes the model's weights and bias, then iteratively updates these parameters by
        moving in the direction of the negative gradient of the loss function (computed using the
        log_likelihood method).

        The regularization terms are added to the gradient of the loss function as follows:

        - No regularization: The standard gradient descent updates are applied without any modification.

        - L1 (Lasso) regularization: Adds a term to the gradient that penalizes the absolute value of
            the weights, encouraging sparsity. The update rule for weight w_j is adjusted as follows:
            dw_j += (C / m) * sign(w_j) - Make sure you understand this!

        - L2 (Ridge) regularization: Adds a term to the gradient that penalizes the square of the weights,
            discouraging large weights. The update rule for weight w_j is:
            dw_j += (C / m) * w_j       - Make sure you understand this!


        - ElasticNet regularization: Combines L1 and L2 penalties.
            The update rule incorporates both the sign and the magnitude of the weights:
            dw_j += l1_ratio * gradient_of_lasso + (1 - l1_ratio) * gradient_of_ridge


        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of examples and n is
                            the number of features.
        - y (np.ndarray): The true labels of the data, with shape (m,).
        - learning_rate (float): The step size at each iteration while moving toward a minimum of the
                            loss function.
        - num_iterations (int): The number of iterations for which the optimization algorithm should run.
        - penalty (str): Type of regularization (None, 'lasso', 'ridge', 'elasticnet'). Default is None.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
                            l1_ratio=0 corresponds to L2 penalty,
                            l1_ratio=1 to L1. Only used if penalty='elasticnet'.
                            Default is 0.5.
        - C (float): Inverse of regularization strength; must be a positive float.
                            Smaller values specify stronger regularization.
        - verbose (bool): Print loss every print_every iterations.
        - print_every (int): Period of number of iterations to show the loss.



        Updates:
        - self.weights: The weights of the model after training.
        - self.bias: The bias of the model after training.
        """

        # TODO: Obtain m (number of examples) and n (number of features)
        m, n = X.shape  
        self.weights = np.zeros(n)
        self.bias = 0.0

        for i in range(num_iterations):
            y_hat = self.predict_proba(X)

            if verbose and i % print_every == 0:
                loss = self.log_likelihood(y, y_hat)
                print(f"Iteration {i}: Loss {loss}")

            dw = np.dot(X.T, (y_hat - y))  
            db = np.sum(y_hat - y) 

            if penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db


    def predict_proba(self, X):
        """
        Predicts probability estimates for all classes for each sample X.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of samples and
            n is the number of features.

        Returns:
        - A numpy array of shape (m, 1) containing the probability of the positive class for each sample.
        """

        # TODO: z is the value of the logits. Write it here (use self.weights and self.bias)
        z = np.dot(X, self.weights) 
        z +=  self.bias
        return self.sigmoid(z)
    def predict(self, X, threshold=0.5):
        """
        Predicts class labels for samples in X.

        Parameters:
        - X (np.ndarray): The input features, with shape (m, n), where m is the number of samples and n
                            is the number of features.
        - threshold (float): Threshold used to convert probabilities into binary class labels.
                            Defaults to 0.5.

        Returns:
        - A numpy array of shape (m,) containing the class label (0 or 1) for each sample.
        """
        # TODO: Predict the class for each input data given the threshold in the argument´
        lista_numeros = []
        probabilities = self.predict_proba(X)
        for i in range(0,len(X)):
            if probabilities[i] > threshold:
                lista_numeros.append(1)
            else:
                lista_numeros.append(0)
        classification_result = np.array(lista_numeros)

        return classification_result

    def lasso_regularization(self, dw, m, C):
        """
        Applies L1 regularization (Lasso) to the gradient during the weight update step in gradient descent.
        L1 regularization encourages sparsity in the model weights, potentially setting some weights to zero,
        which can serve as a form of feature selection.

        The L1 regularization term is added directly to the gradient of the loss function with respect to
        the weights. This term is proportional to the sign of each weight, scaled by the regularization
        strength (C) and inversely proportional to the number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                    Smaller values specify stronger regularization.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
        after applying L1 regularization.
        """

        # TODO:
        # ADD THE LASSO CONTRIBUTION TO THE DERIVATIVE OF THE OBJECTIVE FUNCTION
        lasso_gradient = []
        for i in range(0,len(self.weights)):
            if self.weights[i] > 0:
                lasso_gradient.append(1*C/m)
            elif self.weights[i]< 0:
                lasso_gradient.append(-1*C/m)
            else:
                lasso_gradient.append(0)
        lasso_gradient = np.array(lasso_gradient)
        return dw + lasso_gradient

    def ridge_regularization(self, dw, m, C):
        """
        Applies L2 regularization (Ridge) to the gradient during the weight update step in gradient descent.
        L2 regularization penalizes the square of the weights, which discourages large weights and helps to
        prevent overfitting by promoting smaller and more distributed weight values.

        The L2 regularization term is added to the gradient of the loss function with respect to the weights
        as a term proportional to each weight, scaled by the regularization strength (C) and inversely
        proportional to the number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                    Smaller values specify stronger regularization.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
                        after applying L2 regularization.
        """

        # TODO:
        # ADD THE RIDGE CONTRIBUTION TO THE DERIVATIVE OF THE OBJECTIVE FUNCTION
        ridge = [(self.weights[i]*C)/m for i in range(len(self.weights))]
        ridge = np.array(ridge)
        return dw + ridge

    def elasticnet_regularization(self, dw, m, C, l1_ratio):
        """
        Applies Elastic Net regularization to the gradient during the weight update step in gradient descent.
        Elastic Net combines L1 and L2 regularization, incorporating both the sparsity-inducing properties
        of L1 and the weight shrinkage effect of L2. This can lead to a model that is robust to various types
        of data and prevents overfitting.

        The regularization term combines the L1 and L2 terms, scaled by the regularization strength (C) and
        the mix ratio (l1_ratio) between L1 and L2 regularization. The term is inversely proportional to the
        number of samples (m).

        Parameters:
        - dw (np.ndarray): The gradient of the loss function with respect to the weights, before regularization.
        - m (int): The number of samples in the dataset.
        - C (float): Inverse of regularization strength; must be a positive float.
                     Smaller values specify stronger regularization.
        - l1_ratio (float): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds
                            to L2 penalty, l1_ratio=1 to L1. Only used if penalty='elasticnet'.
                            Default is 0.5.

        Returns:
        - np.ndarray: The adjusted gradient of the loss function with respect to the weights,
                      after applying Elastic Net regularization.
        """
        # TODO:
        # ADD THE RIDGE CONTRIBUTION TO THE DERIVATIVE OF THE OBJECTIVE FUNCTION
        # Be careful! You can reuse the previous results and combine them here, but beware how you do this!
        ridge_gradient = (C / m) * np.array(self.weights)

        lasso_gradient = []
        for i in range(0,len(self.weights)):
            if self.weights[i] > 0:
                lasso_gradient.append(1*C/m)
            elif self.weights[i]< 0:
                lasso_gradient.append(-1*C/m)
            else:
                lasso_gradient.append(0)
        lasso_gradient = np.array(lasso_gradient)
        
        elasticnet_gradient = (1-l1_ratio)*ridge_gradient + l1_ratio*lasso_gradient
        return dw + elasticnet_gradient

    @staticmethod
    def log_likelihood(y, y_hat):
        """
        Computes the Log-Likelihood loss for logistic regression, which is equivalent to
        computing the cross-entropy loss between the true labels and predicted probabilities.
        This loss function is used to measure how well the model predicts the actual class
        labels. The formula for the loss is:

        L(y, y_hat) = -(1/m) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))

        where:
        - L(y, y_hat) is the loss function,
        - m is the number of observations,
        - y is the actual label of the observation,
        - y_hat is the predicted probability that the observation is of the positive class,
        - log is the natural logarithm.

        Parameters:
        - y (np.ndarray): The true labels of the data. Should be a 1D array of binary values (0 or 1).
        - y_hat (np.ndarray): The predicted probabilities of the data belonging to the positive class (1).
                            Should be a 1D array with values between 0 and 1.

        Returns:
        - The computed loss value as a scalar.
        """

        # TODO: Implement the loss function (log-likelihood)
        e = 1e-15  
        m = len(y) 
        loss = 0 
        for i in range(m):
            if y_hat[i] < e:
                y_hat[i] = e
            elif y_hat[i] > 1 - e:
                y_hat[i] = 1 - e
            loss += y[i] * np.log(y_hat[i]) + (1 - y[i]) * np.log(1 - y_hat[i])
        loss = -loss / m  
        return loss
    
    
    @staticmethod
    def sigmoid(z):
        """
        Computes the sigmoid of z, a scalar or numpy array of any size. The sigmoid function is used as the
        activation function in logistic regression, mapping any real-valued number into the range (0, 1),
        which can be interpreted as a probability. It is defined as 1 / (1 + exp(-z)), where exp(-z)
        is the exponential of the negative of z.

        Parameters:
        - z (float or np.ndarray): Input value or array for which to compute the sigmoid function.

        Returns:
        - The sigmoid of z.
        """

        # TODO: Implement the sigmoid function to convert the logits into probabilities
        if isinstance(z,np.ndarray): 
            r = np.zeros_like(z)  
            for i in range(len(z)):  
                r[i] = 1/(1+np.exp(-z[i]))  
            return r
        else:  
            return 1/(1+np.exp(-z))