import numpy as np

class LassoHomotopyModel:
    """
    LASSO regularized regression model using the Homotopy Method with online updates.
    Implements the algorithm from "An Homotopy Algorithm for the Lasso with Online Observations".
    """
    
    def __init__(self, mu=None, tol=1e-6, max_iter=1000):
        self.mu = mu  # Regularization parameter
        self.tol = tol
        self.max_iter = max_iter
        self.coef_ = None
        self.active_set = []
        self.signs = []
        self.X_seen = None
        self.y_seen = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = 0

    def _standardize(self, X, y=None):
        """Standardize features and center target."""
        if self.X_mean is None:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std[self.X_std == 0] = 1  # Avoid division by zero
            if y is not None:
                self.y_mean = y.mean()
        
        X = (X - self.X_mean) / self.X_std
        if y is not None:
            y = y - self.y_mean
        return X, y if y is not None else X

    def fit(self, X, y):
        """Fit model on dataset."""
        X, y = self._standardize(X, y)
        self.X_seen = X
        self.y_seen = y
        n_samples, n_features = X.shape
        
        if self.mu is None:
            self.mu = 0.1 * np.max(np.abs(X.T @ y))
        
        # Initialize coefficients and active set
        self.coef_ = np.zeros(n_features)
        self.active_set = []
        self.signs = []
        
        # Homotopy path for fit
        self._compute_homotopy(X, y, mu_start=self.mu)
        return self

    def _compute_homotopy(self, X, y, mu_start):
        """Core homotopy algorithm from paper."""
        n_samples, n_features = X.shape
        mu = mu_start
        theta = np.zeros(n_features)
        active_set = []
        signs = []
        
        for _ in range(self.max_iter):
            residual = y - X @ theta
            correlations = X.T @ residual
            
            # Check optimality conditions
            violating = np.where(np.abs(correlations) > mu + self.tol)[0]
            if len(violating) == 0:
                break
                
            # Add most violating feature to active set
            j = violating[np.argmax(np.abs(correlations[violating]))]
            sign = np.sign(correlations[j])
            
            if j not in active_set:
                active_set.append(j)
                signs.append(sign)
                
            # Solve for active coefficients
            X_active = X[:, active_set]
            try:
                beta = np.linalg.pinv(X_active.T @ X_active) @ (X_active.T @ y - mu * np.array(signs))
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X_active, y, rcond=None)[0] - mu * np.array(signs) / (X_active.T @ X_active).diagonal()
            
            theta = np.zeros(n_features)
            theta[active_set] = beta
            
            # Check for coefficients leaving active set
            zero_coeffs = np.where(np.abs(beta) < self.tol)[0]
            if len(zero_coeffs) > 0:
                removed = active_set[zero_coeffs[0]]
                active_set.pop(zero_coeffs[0])
                signs.pop(zero_coeffs[0])
                
            # Update regularization parameter
            mu = self.mu  
            
        self.coef_ = theta
        self.active_set = active_set
        self.signs = signs

    def update(self, X_new, y_new):
        """Update model with new observation using homotopy continuation."""
        if self.coef_ is None:
            return self.fit(X_new, y_new)
            
        # Standardize new data point
        X_new, y_new = self._standardize(X_new.reshape(1, -1), y_new - self.y_mean)
        
        # Update dataset
        self.X_seen = np.vstack([self.X_seen, X_new])
        self.y_seen = np.append(self.y_seen, y_new)
        
        # Homotopy parameters
        t_values = np.linspace(0, 1, 10)
        theta = self.coef_.copy()
        active_set = self.active_set.copy()
        signs = self.signs.copy()
        
        for t in t_values:
            # Form augmented data with weight t
            X_aug = np.vstack([self.X_seen[:-1], t * X_new])
            y_aug = np.append(self.y_seen[:-1], t * y_new)
            
            # Update homotopy
            self._update_homotopy_step(X_aug, y_aug, theta, active_set, signs, t)
            
        self.coef_ = theta
        self.active_set = active_set
        self.signs = signs
        return self

    def _update_homotopy_step(self, X, y, theta, active_set, signs, t):
        """Compute one step of homotopy update."""
        residual = y - X @ theta
        correlations = X.T @ residual
        mu = self.mu
        
        # Check for violations in KKT conditions
        violating_inactive = [
            j for j in range(X.shape[1]) 
            if j not in active_set and np.abs(correlations[j]) > mu + self.tol
        ]
        
        # Add violating features
        if violating_inactive:
            j = violating_inactive[np.argmax(np.abs(correlations[violating_inactive]))]
            sign = np.sign(correlations[j])
            active_set.append(j)
            signs.append(sign)
            
        # Solve for active coefficients
        if active_set:
            X_active = X[:, active_set]
            try:
                beta = np.linalg.pinv(X_active.T @ X_active) @ (X_active.T @ y - mu * np.array(signs))
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X_active, y, rcond=None)[0] - mu * np.array(signs) / (X_active.T @ X_active).diagonal()
            
            theta[:] = 0
            theta[active_set] = beta
            
            # Remove zero coefficients
            zero_coeffs = np.where(np.abs(beta) < self.tol)[0]
            for idx in reversed(sorted(zero_coeffs)):
                active_set.pop(idx)
                signs.pop(idx)
                
        return theta
    def predict(self, X):
        """
        Make predictions using the learned model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        if self.coef_ is None:
            raise ValueError("Model has not been fitted yet!")
            
        # Convert to numpy array and ensure 2D shape
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Standardize using training parameters
        X_std = (X - self.X_mean) / self.X_std
        
        # Make prediction and add back y mean
        return X_std @ self.coef_ + self.y_mean

class LassoHomotopyResults:
    def __init__(self, coef=None, active_set=None):
        self.coef_ = coef
        self.active_set = active_set

    def predict(self, X):
        return X @ self.coef_