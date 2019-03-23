from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

"""
    Training classifiers, hyper parameters for each model is set without tuning
"""
def train(X_train, y_train):
    classifiers = [SVC(C=1, gamma=1), SVC(kernel='linear', C=0.1),
                    MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, alpha=0.01, tol=0.01),
                    RandomForestClassifier(n_estimators=20, max_depth=5, max_features=1)]
    names = ['RBF SVM', 'Linear SVM', 'Neural Network', 'Random Forest']
    for i, clf in enumerate(classifiers):
        print('Training ', names[i])
        clf.fit(X_train, y_train)
    return zip(names, classifiers)