import pandas as pd                                                         # For loading the DataFrame
from sklearn import preprocessing                                           # For normalising the data
from sklearn.model_selection import train_test_split, cross_val_score       # For splitting the data
from sklearn.tree import DecisionTreeClassifier                             # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier                         # Random Forest Classifier
from sklearn import svm                                                     # Support Vector Machine Classifier
from sklearn import neighbors                                               # K Nearest Neighbours Classifier
from sklearn.naive_bayes import MultinomialNB                               # Naive Bayes Classifier
from sklearn.linear_model import LogisticRegression                         # Logistic Regression Classifier

# For the Neural Network
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_csv('data/mammographic_masses.data.txt', names=['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'], na_values = ['?'])
df.dropna(inplace=True) # Cleaning the data

# Converting DataFrame to Arrays
feature_labels = ['Age', 'Shape', 'Margin', 'Density']
features = df[feature_labels].values
classes = df['Severity'].values

# Normalising the data
normaliser = preprocessing.StandardScaler()
scaled_features = normaliser.fit_transform(features)

# Splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, classes, test_size=0.30)

# 1. CLASSIFICATION USING DECISION TREES
dec_tree = DecisionTreeClassifier(random_state=1)
dec_tree.fit(X_train, y_train)
decision_tree_score = dec_tree.score(X_test, y_test) * 100

# 2. CLASSIFICATION USING RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=10, random_state=1)
cv_score = cross_val_score(random_forest, scaled_features, classes, cv=10)
random_forest_score = cv_score.mean() * 100

# 3. CLASSIFICATION USING SUPPORT VECTOR MACHINE (SVM)
C = 1.0

# Linear kernel
svc_linear = svm.SVC(kernel='linear', C=C)
svc_linear_cv_score = cross_val_score(svc_linear, scaled_features, classes, cv=10)
svc_linear_score = svc_linear_cv_score.mean() * 100

# Polynomial kernel
svc_polynomial = svm.SVC(kernel='poly', C=C)
svc_polynomial_cv_score = cross_val_score(svc_polynomial, scaled_features, classes, cv=10)
svc_polynomial_score = svc_polynomial_cv_score.mean() * 100

# RBF kernel
svc_rbf = svm.SVC(kernel='rbf', C=C)
svc_rbf_cv_score = cross_val_score(svc_rbf, scaled_features, classes, cv=10)
svc_rbf_score = svc_rbf_cv_score.mean() * 100

# 4. CLASSIFICATION USING K NEAREST NEIGHBOURS
# Looping from K = 1 to 30, returning the best val
all_knn = []
for i in range(1, 30):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn_cv_score = cross_val_score(knn, scaled_features, classes, cv=10)
    all_knn.append(knn_cv_score.mean())
knn_score = max(all_knn) * 100
k = all_knn.index(knn_score/100) + 1

# 5. CLASSIFICATION USING NAIVE BAYES
nb_normaliser = preprocessing.MinMaxScaler()
nb_features = nb_normaliser.fit_transform(features)
nb = MultinomialNB()
nb_cv_score = cross_val_score(nb, nb_features, classes, cv=10)
nb_score = nb_cv_score.mean() * 100

# 6. CLASSIFICATION USING LOGISTIC REGRESSION
log_reg = LogisticRegression()
log_reg_cv_score = cross_val_score(log_reg, scaled_features, classes, cv=10)
log_reg_score = log_reg_cv_score.mean() * 100

# 7. CLASSIFICATION USING A NEURAL NETWORK
def create_model():
    model = Sequential()
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model_estimator = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)
neural_network = cross_val_score(model_estimator, scaled_features, classes, cv=10)
neural_network_score = neural_network.mean() * 100

print("CLASSIFICATION ACCURACY RESULTS:")
print(f'Decision Trees = {decision_tree_score} %')
print(f'Random Forest = {random_forest_score} %')
print(f'SVM linear kernel = {svc_linear_score} %')
print(f'SVM polynomial kernel = {svc_polynomial_score} %')
print(f'SVM rbf kernel = {svc_rbf_score} %')
print(f'KNN = {knn_score} % for k = {k}')
print(f'Naive Bayes = {nb_score} %')
print(f'Logistic Regression = {log_reg_score} %')
print(f'Neural Network = {neural_network_score} %')

