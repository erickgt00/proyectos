## Predicting the Delinquency of Clients with Bank Credits Using Machine Learning

### Abstract
In this project, prediction models are made to find out if a client will be delinquent in relation to different variables such as the amount of the loan, the term, among others. This helps us to anticipate if a client will default on payment and thus take action in this regard.
Code and PDF:

> [Pyhton Code](https://github.com/erickgt00/proyectos/blob/main/Credit%20Default%20ML/ARTICULO.ipynb)

> [PDF: Predicción de la Morosidad de Clientes con Créditos Bancarios Utilizando Aprendizaje Automático](https://github.com/erickgt00/proyectos/blob/main/Credit%20Default%20ML/ANALISIS%20DE%20CREDITO.pdf)


Machine Learning methods:

* Logistic regression.
* Decision tree.
* Random Forest.
* K Nearest Neighbors.

Librarys:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score,recall_score, f1_score,confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression,RidgeClassifier
import xgboost as xgb

```
