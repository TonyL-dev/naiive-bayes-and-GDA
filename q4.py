'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    labels = set(train_labels)
    for c in labels:
      current_x = train_data[train_labels == c]
      means[c.astype(int)] = current_x.mean(axis=0)

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    N, M = train_data.shape

    labels = set(train_labels)
    for c in labels:
      #deviation_score = X - 1*1'*X*(1/N)
      #variance_covariance = deviation_score*deviation_score'*(1/N)
      current_x = train_data[train_labels == c]
      N, M = current_x.shape
      ones = np.ones((N,1))

      ones_matrix = ones * ones.transpose()

      dev_score_subtract = np.matmul(ones_matrix, current_x)
      dev_score_subtract = dev_score_subtract/N

      deviation_score = current_x - dev_score_subtract

      variance_matrix = np.matmul(deviation_score.transpose(), deviation_score)

      variance_matrix = variance_matrix/N
      
      #add 0.01I to matrix
      iden = np.identity(64)
      iden = iden*0.01
      variance_matrix = variance_matrix + iden 

      covariances[c.astype(int)] = variance_matrix
    

    # Compute covariances
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    _,d,__ = covariances.shape
    N = digits.shape[0]
    log_like = np.zeros((N, 10))

    #compute values
    sigma_inv = np.linalg.inv(covariances)
    sigma_det = np.linalg.det(covariances)
    const_term = (-d/2) * np.log(2*np.pi)
    
    for i in range(N):
      for c in range(10):
        sum = const_term
        exponent_term = -(1/2) * ((digits[i]-means[c]).T @ sigma_inv[c] @ (digits[i] - means[c]))
        det_term = -(1/2) * np.log(sigma_det[c])
        sum = sum + exponent_term + det_term
        log_like[i, c] = sum
    return log_like

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    condi_likelihood = generative_likelihood(digits, means, covariances)
    condi_likelihood = condi_likelihood + np.log(0.1)
    return condi_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    sum = 0
    for i in range(digits.shape[0]):
      sum += cond_likelihood[i, labels[i].astype(int)]
    sum = sum/digits.shape[0]
    # Compute as described above and return
    return sum

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    predictions = np.argmax(condi_likelihood, axis = 1)
    return predictions

def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)

def predict(digits, labels, means, covariances):
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    predictions = np.argmax(cond_likelihood, axis=1)
    return predictions


def accuracy(predictions, labels):
    N = labels.shape[0]
    labels.reshape(N, 1)

    acc = np.count_nonzero(predictions == labels)/N
    return acc

def main():
    train_data, train_labels, test_data, test_labels = load_all_data('data')

    # Fit the model - training
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    avg_cond_likeli = avg_conditional_likelihood(train_data, train_labels,means, covariances)
    
    print("Average conditional likelihood on the training set is ", avg_cond_likeli)
    # Evaluation - training
    predictions = predict(train_data, train_labels, means, covariances)
    acc = accuracy(predictions, train_labels)
    print("Training accuracy for conditional likelihood is ", acc)
    
    # Evaluation - test
    avg_cond_likeli = avg_conditional_likelihood(test_data, test_labels,means, covariances)
    
    print("Average conditional likelihood on the test set is ", avg_cond_likeli)

    predictions = predict(test_data, test_labels, means, covariances)
    
    acc = accuracy(predictions, test_labels)
    print("Testing accuracy for conditional likelihood is ", acc)


    # Plotting images for test set
    fig = plt.figure(figsize = (8, 8))
    rows = 2
    columns = 5
    for c in range(10):
      eigval, eigvect = np.linalg.eig(covariances[c])
      max_val = np.argmax(eigval, axis = 0)
      fig.add_subplot(rows, columns, c+1)
      lead_eigvect = eigvect[:,max_val]
      lead_eigvect = lead_eigvect.reshape(8,8)
      plt.imshow(lead_eigvect)
    plt.show()


if __name__ == '__main__':
    main()