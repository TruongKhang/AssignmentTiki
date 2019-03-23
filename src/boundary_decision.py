from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset import *
from classifier import train

"""
    calculate the probability of data points for class 1
    Input:
        points: array of data points, each point is (x, y)
        weights0: weights of mixed gaussian distribution for class 0 (mixed distribution 0)
        list_gauss0: components of mixed distribution 0
        weights: weights of mixed distribution 1
        list_gauss1: components of mixed distribution 1
    Return:
        array of probability score

"""
def prob_C1(points, weights0, list_gauss0, weights1, list_gauss1):
    list_prob = list()
    for point in points:
        pdf_C0 = mixed_gauss_dist(point, weights0, list_gauss0)
        pdf_C1 = mixed_gauss_dist(point, weights1, list_gauss1)
        list_prob.append(pdf_C1 / (pdf_C0 + pdf_C1))
    return np.array(list_prob)


if __name__ == '__main__':
    dist0, dist1 = generate_data()
    data0, weights0, list_gauss0 = dist0
    data1, weights1, list_gauss1 = dist1 
    inputs = np.concatenate((data0, data1), axis=0)
    labels = np.append([0]*len(data0), [1]*len(data1))

    # Draw boundary decision as below

    # create a grid [x_min, x_max] x [y_min, y_max]
    x_min, x_max = min(inputs[:, 0]), max(inputs[:, 0])
    y_min, y_max = min(inputs[:, 1]), max(inputs[:, 1])
    grid_x = np.arange(x_min-0.02, x_max+0.02, 0.02)
    grid_y = np.arange(y_min-0.02, x_max+0.02, 0.02)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, test_size=0.2, stratify=labels)
   
    figure = plt.figure(figsize=(27,9))

    """
        Draw decision boundary without training a ML model (Baysian boundary decision)
    """
    z = prob_C1(grid_points, weights0, list_gauss0, weights1, list_gauss1).reshape(xx.shape) # calculate probability score for each point in grid
    ax = plt.subplot(2,3,1)
    color = ['r', 'b']
    for i, data in enumerate([data0, data1]):
        label = color[i]
        for point in data: 
            ax.scatter(point[0], point[1], c=label) # plot data points
    # plot decision boundary of generative model
    # points with probability score < 0.5 (or >= 0.5) have the same color
    ax.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=.8, levels=[0, 0.5, 1])
    # calculate accuracy
    prob_test = prob_C1(inputs, weights0, list_gauss0, weights1, list_gauss1)
    pred_test = np.where(prob_test > 0.5, 1, 0)
    score = accuracy_score(labels, pred_test)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.text(x_max - .3, y_min + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
    ax.set_title('Generative Model')
    
    id_fig = 2
    """
        Draw decision boundary by training a ML model
        Here, the models are: Linear SVM, SVM with gauss kernel (RBF), Neural Network and Random Forest
    """
    # Training a ML model
    for name, clf in train(inputs, labels):
        # calculate probability score for each point in grid, each model has its own probability
        if name in ['Linear SVM', 'RBF SVM']:
            z = clf.decision_function(grid_points)
            levels = [z.min(), 0, z.max()]
        else:
            z = clf.predict_proba(grid_points)[:, 1]
            levels = [0, 0.5, 1]
        z = z.reshape(xx.shape)
        ax = plt.subplot(2,3,id_fig)
        color = ['r', 'b']
        for i, data in enumerate([data0, data1]):
            label = color[i]
            for point in data: 
                ax.scatter(point[0], point[1], c=label) # plot data points
        # plot decision boundary of trained model
        ax.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=.8, levels=levels)
        # calculate accuracy
        score = clf.score(inputs, labels)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.text(x_max - .3, y_min + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        ax.set_title(name)
        id_fig += 1
    plt.tight_layout()
    plt.show()
    
