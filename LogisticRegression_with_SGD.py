import numpy as np
from numpy import random
import matplotlib.pyplot as plt



def read_data():

    train_data = np.load('./train_features.npy')
    train_labels = np.load('./train_labels.npy')
    test_data = np.load('./test_features.npy')
    test_labels = np.load('./test_labels.npy')

    return train_data, train_labels, test_data, test_labels

def learning_algorithm(X, Y, learning_rate, epochs):
    
    t = 0
    W = np.zeros(X.shape[1]+1)
    loss_values = []
    learning_rates = []
    current_loss = 1e2
    previous_loss = 0
    
    while(abs(current_loss - previous_loss) > 1e-5):
    
        for epoch in range(epochs):

            suff = np.arange(X.shape[0])
            np.random.shuffle(suff)
            X_shuffle = X[suff]
            Y_shuffle = Y[suff]
            X_shuffle = add_intercept(X_shuffle)

            for i in range(len(X_shuffle)):
                gt = gradient(X_shuffle[i],Y_shuffle[i],W)
                W = W -gt * learning_rate

            loss_value = loss_function(X_shuffle,Y_shuffle,W)
            loss_values.append(loss_value)
            learning_rates.append(learning_rate)
            learning_rate = learning_rate * (1 / (1 + 1e-4 * epochs))   
            previous_loss = current_loss
            current_loss = loss_value
            t = t+1

    return W, loss_values, learning_rates

def loss_function(X,Y,W):

    total_loss = 0
    N = X.shape[0]
    
    for i in range(0,N):
        dot = np.dot(X[i],W)
        total_loss = total_loss + np.log(1+np.exp(-Y[i]*dot))

    loss = total_loss / N
    
    return loss

def add_intercept(X):
    
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)

def gradient(X,Y,W):
    
    dot = np.dot(X,W)
    grad = -((Y*X) / (1+np.exp(Y*dot)))
        
    return grad

def accuracy(data,weight,labels):
       
        counter = 0
        threshold = 0.5
        dot_product = np.dot(data,weight)
        prediction = 1 / (1 + np.exp(-dot_product))  #sigmoid function
        
        for i in range(0,len(prediction)):
            
            if(prediction[i]>=threshold and labels[i]==1):
                    counter = counter+1    
            elif(prediction[i]<threshold and labels[i]==-1):
                    counter = counter+1
                    
        result = (counter / len(prediction))*100
        
        return result


def scatter_plot(data, label):
    
    for i in range(0,len(data)):
        color = ""
        if(label[i] == 1):
            color = "blue"
        else:
            color = "red"

        x = data[i][0]
        y = data[i][1]
        plt.scatter(x,y,c=color)

def convergence_plot(loss_values):

    x = np.arange(len(loss_values))
    y = loss_values
    plt.plot(x, y)
    plt.title('Convergence Curve')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss Values')
    plt.show()

def decision_boundary_plot(data, labels, weights, title):

    plt.figure()
    scatter_plot(data,labels)
    ax = plt.gca()
    ax.autoscale(False)
    X = np.array(ax.get_xlim())
    Y = -(weights[0]+weights[1]*X)/weights[2]
    plt.plot(X, Y, '--', c="black", label='Decision Boundary')
    plt.title(title)
    plt.legend()
    plt.show()

def main():

    train_data, train_labels, test_data, test_labels = read_data()
    X = add_intercept(train_data)
    Y = add_intercept(test_data)
    learning_rate = 0.01
    epochs = 100
    final_W, loss_values, learning_rates = learning_algorithm(train_data, train_labels, learning_rate, epochs)
    training_accuracy = accuracy(X, final_W, train_labels)
    print(f'Training accuracy is {training_accuracy}')
    test_accuracy = accuracy(Y, final_W, test_labels)
    print(f'Test accuracy is {test_accuracy}')
    convergence_plot(loss_values)
    # decision_boundary_plot(train_data, train_labels, final_W, 'Training Data')
    # decision_boundary_plot(test_data, test_labels, final_W, 'Test Data')


if __name__ == '__main__':
    main()

