from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

iris = datasets.load_iris()
x = iris.data
y = iris.target_names[iris.target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

lng = svm.SVC()
lng.fit(x_train, y_train)

lng.score(x_test, y_test)

def model(dat):
    pred = lng.predict(dat)
    return pred
model(x_test)

iris = datasets.load_iris()
dat01 = pd.DataFrame(iris.data, columns=['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)'])
dat02 = pd.DataFrame(iris.target_names[iris.target], columns=['species'])
df = pd.concat([dat01, dat02], axis=1)

# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['species'] = iris.target_names[iris.target]

sns.pairplot(dat, hue='species')