import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.calibration import LabelEncoder

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
import missingno as msno
import pickle

import warnings
warnings.simplefilter(action="ignore")

df = pd.read_csv("./diabetes.csv")
df.head()


def check_df(dataframe,head=5):
  print("######################### Head #########################")
  print(dataframe.head(head))
  print("######################### Tail #########################")
  print(dataframe.tail(head))
  print("######################### Shape #########################")
  print(dataframe.shape)
  print("######################### Types #########################")
  print(dataframe.dtypes)
  print("######################### NA #########################")
  print(dataframe.isnull().sum())
  print("######################### Qurtiles #########################")
  print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
  
  
check_df(df)



def grab_col_names(dataframe, cat_th=10, car_th=20):
  #Catgeorical Variable Selection
  cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category","object","bool"]]
  num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["uint8","int64","float64"]]
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category","object"]]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  #Numerical Variable Selection
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["uint8","int64","float64"]]
  num_cols = [col for col in num_cols if col not in cat_cols]

  return cat_cols, num_cols, cat_but_car



cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Print Categorical and Numerical Variables
print(f"Observations: {df.shape[0]}")
print(f"Variables: {df.shape[1]}")
print(f"Cat_cols: {len(cat_cols)}")
print(f"Num_cols: {len(num_cols)}")
print(f"Cat_but_car: {len(cat_but_car)}")



def cat_summary(dataframe,col_name,plot=False):
  print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                      'Ration': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
  print("##########################################")
  if plot:
    sns.countplot(x=dataframe[col_name],data=dataframe)
    plt.show(block=True)
    
    
def cat_summary_df(dataframe):
  cat_cols, num_cols, cat_but_car = grab_col_names(df)
  for col in cat_cols:
    cat_summary(dataframe, col, plot=True)
    
    
cat_summary_df(df)



def num_summary(dataframe, num_col, plot=False):
  quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
  print(dataframe[num_col].describe(quantiles).T)

  if plot:
    dataframe[num_col].hist(bins=20)
    plt.xlabel(num_col)
    plt.title(num_col)
    plt.show(block=True)
    
    
def num_summary_df(dataframe):
  cat_cols, num_cols, cat_but_car = grab_col_names(df)
  for col in num_cols:
    num_summary(dataframe, col, plot=True)
    
    
num_summary_df(df)


def plot_num_summary(dataframe):
  cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
  plt.figure(figsize=(12,4))
  for index, col in enumerate(num_cols):
    plt.subplot(2,4,index+1)
    plt.tight_layout()
    dataframe[col].hist(bins=20)
    plt.title(col)
    
plot_num_summary(df)


def target_summary_with_num(dataframe, target, numerical_col):
  print(dataframe.groupby(target).agg({numerical_col: "mean"}))
  print("#############################################")
  
def target_summary_with_num_df(dataframe, target):
  cat_cols, num_cols, cat_but_car = grab_col_names(df)
  for col in num_cols:
    target_summary_with_num(dataframe, target, col)
    
target_summary_with_num_df(df, "Outcome")


df.corr()



f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()



def high_correlated_cols(dataframe, plot=False, corr_th = 0.90):
  corr = dataframe.corr()
  corr_matrix = corr.abs()
  upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
  drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

  if drop_list == []:
    print("############## After Correlation Analysis, You Don't Need to Remove Variables ##############")

  if plot:
    sns.set(rc = {'figure.figsize':(18,13)})
    sns.heatmap(corr, cmap="RdBu")
    plt.show()
  return drop_list



high_correlated_cols(df, plot=True)





def exploratory_data(dataframe):
  import warnings
  warnings.filterwarnings('ignore')
  cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
  fig,ax = plt.subplots(8,3,figsize=(30,90))
  for index, col in enumerate(num_cols):
    sns.distplot(dataframe[col],ax=ax[index,0])
    sns.boxplot(dataframe[col],ax=ax[index,1])
    stats.probplot(dataframe[col],plot=ax[index,2])
  fig.tight_layout()
  fig.subplots_adjust(top=0.95)
  plt.suptitle("Visualizing Continuous Columns")
  
  
  
  exploratory_data(df)
  
  df.isnull().sum()
  
  
  
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
for col in zero_columns:
  df[col] = np.where(df[col]==0, np.nan, df[col])
  
  
  
df.isnull().sum() 



def missing_value_table(dataframe, na_name=False):
  na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
  n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
  ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
  missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss","ratio"])
  print(missing_df, end="\n")
  if na_name:
    return na_columns

na_columns = missing_value_table(df, na_name=True)


def show_missing_value_plot(dataframe, bar=True, matrix=True, heatmap=True):
  if bar:
    msno.bar(dataframe);
  if matrix:
    msno.matrix(dataframe);
  if heatmap:
    msno.heatmap(dataframe);
    
    
    
show_missing_value_plot(df)



def missing_vs_target(dataframe, target):
  na_columns = missing_value_table(dataframe, na_name=True)
  temp_df = dataframe.copy()
  for col in na_columns:
    temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
  na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
  for col in na_flags:
    print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                        "Count": temp_df.groupby(col)[target].count()}))
    print("##################################################")
    
missing_vs_target(df, "Outcome")



def quick_missing_imp(data, target, num_method="meidan", cat_length=20):
  variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]
  temp_target = data[target]

  print("# BEFORE")
  print(data[variables_with_na].isnull().sum(), "\n\n")

  data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)
  if num_method == "mean":
      data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
  elif num_method == "median":
      data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
  data[target] = temp_target

  print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
  print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
  print(data[variables_with_na].isnull().sum(), "\n\n")

  return data




df = quick_missing_imp(df, "Outcome", num_method="median", cat_length=17)


df.isnull().sum()


def outlier_thresholds(dataframe,col_name,q1=0.10,q3=0.90):
  quartile1 = dataframe[col_name].quantile(q1)
  quartile3 = dataframe[col_name].quantile(q3)
  interquartile_range = quartile3 - quartile1
  low_limit = quartile1 - 1.5 * interquartile_range
  up_limit = quartile3 + 1.5 * interquartile_range
  return low_limit,up_limit



def check_outlier(dataframe, col_name):
  low_limit,up_limit = outlier_thresholds(dataframe,col_name)
  if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
    return True
  else:
    return False


def replace_with_thresholds(dataframe, col_name):
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
  dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
  
  
  
def solve_outliers(dataframe, target):
  cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
  for col in num_cols:
    if col!=target:
      print(col, check_outlier(dataframe, col))
      if check_outlier(dataframe, col):
        replace_with_thresholds(dataframe, col)
        
        
solve_outliers(df, "Outcome")





def check_outlier_df(dataframe, target):
  cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
  for col in num_cols:
    if col!=target:
      print(col, check_outlier(dataframe, col))
      
      
check_outlier_df(df, "Outcome")


def label_encoder(dataframe, binary_col):
  labelencoder = LabelEncoder()
  dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
  return dataframe



def label_encoder_dataframe(dataframe):
  binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in ["int", "float"] and dataframe[col].nunique() == 2]
  for col in binary_cols:
    label_encoder(dataframe, col)
    
    
label_encoder_dataframe(df)

df.head()


def one_hot_encoding(dataframe, drop_first=True):
  label_encoder_dataframe(dataframe)
  cat_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
  dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
  return dataframe


df = one_hot_encoding(df)


df.head()




def Create_and_Train_Classification_Models(dataframe, target, test_size=0.20, cv=10, plot=False, save_results=False):
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    results_dict = {'Model_Names': [],
                    'ACC_Train': [],
                    'ACC_Test': [],
                    'ACC_All': [],
                    'R2': [],
                    'R2_Train': [],
                    'R2_Test': [],
                    'CV_Train': [],
                    'CV_Test': [],
                    'CV_All': [],
                    'Accuracy': [],
                    'Precision': [],
                    'Recall': [],
                    'F1-Score': [],
                    'Roc_Auc': []
                    }

    models = [('Logistic', LogisticRegression(solver="liblinear")),
              ("Naive", GaussianNB()),
              ("KNN", KNeighborsClassifier()),
              ("SVC", SVC(kernel="rbf", probability=True)),
              ('CART', DecisionTreeClassifier()),
              ('RF', RandomForestClassifier()),
              ("AdaBoost", AdaBoostClassifier()),
              ('BGTrees', BaggingClassifier(bootstrap_features=True)),
              ('GBM', GradientBoostingClassifier())]

    print("###################### Model Results 1######################")

    for name, classifier in models:
        classifier.fit(X_train, y_train)
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)

        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        acc_all = accuracy_score(y, classifier.predict(X))
        r2 = classifier.score(X, y)
        r2_train = classifier.score(X_train, y_train)
        r2_test = classifier.score(X_test, y_test)
        cv_train = cross_val_score(classifier, X_train, y_train, cv=cv).mean()
        cv_test = cross_val_score(classifier, X_test, y_test, cv=cv).mean()
        cv_all = cross_val_score(classifier, X, y, cv=cv).mean()
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

        results_dict['Model_Names'].append(name)
        results_dict['ACC_Train'].append(acc_train)
        results_dict['ACC_Test'].append(acc_test)
        results_dict['ACC_All'].append(acc_all)
        results_dict['R2'].append(r2)
        results_dict['R2_Train'].append(r2_train)
        results_dict['R2_Test'].append(r2_test)
        results_dict['CV_Train'].append(cv_train)
        results_dict['CV_Test'].append(cv_test)
        results_dict['CV_All'].append(cv_all)
        results_dict['Accuracy'].append(cv_results['test_accuracy'].mean())
        results_dict['Precision'].append(cv_results['test_precision'].mean())
        results_dict['Recall'].append(cv_results['test_recall'].mean())
        results_dict['F1-Score'].append(cv_results['test_f1'].mean())
        results_dict['Roc_Auc'].append(cv_results['test_roc_auc'].mean())

        with open(f'{name}_model.pkl', 'wb') as file:
            pickle.dump(classifier, file)

    model_results = pd.DataFrame(results_dict).set_index("Model_Names")
    model_results = model_results.sort_values(by="Accuracy", ascending=False)
    print(model_results)

    if plot:
        plt.figure(figsize=(15, 12))
        sns.barplot(x='Accuracy', y=model_results.index, data=model_results, color="r")
        plt.xlabel('Accuracy Values')
        plt.ylabel('Model Names')
        plt.title('Accuracy for All Models')
        plt.show()

    if save_results:
        model_results.to_csv("model_results.csv")

    return model_results




model_results = Create_and_Train_Classification_Models(df, "Outcome", plot=True, save_results=True)



def Create_and_Train_Classification_Model_Tuning(dataframe, target, test_size=0.20, cv=10, plot=False, save_results=False):
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    results_dict = {'Model_Names': [],
                    'ACC_Train': [],
                    'ACC_Test': [],
                    'ACC_All': [],
                    'R2': [],
                    'R2_Train': [],
                    'R2_Test': [],
                    'CV_Train': [],
                    'CV_Test': [],
                    'CV_All': [],
                    'Accuracy': [],
                    'Precision': [],
                    'Recall': [],
                    'F1-Score': [],
                    'Roc_Auc': [],
                    'Best_Params': []
                    }
    logistic_params = {}
    naive_params = {}
    knn_params = {"n_neighbors": np.arange(2,50)}
    svc_params = {"C": [0.1,0.01,0.001,1,5,10,20,50,100],
                        "gamma": [0.1,0.01,0.001,1,5,10,20,50,100]}
    cart_params = {"min_samples_split": range(2,100),
                   "max_leaf_nodes": range(2,10)}
    adaboost_params = {"learning_rate": [0.01, 0.1],
                       "n_estimators": [100, 500, 1000]}
    
    rf_params = {"max_depth": [5, 8, 15, None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [8, 15, 20],
                 "n_estimators": [200, 500]}
    gbm_params = {"learning_rate": [0.01, 0.1],
                  "max_depth": [3, 8],
                  "n_estimators": [500, 1000],
                  "subsample": [1, 0.5, 0.7]}

    models = [('Logistic', LogisticRegression(solver="liblinear"), logistic_params),
              ("Naive", GaussianNB(), naive_params),
              ("KNN", KNeighborsClassifier(), knn_params),
              ("SVC", SVC(kernel="rbf", probability=True), svc_params),
              ('CART', DecisionTreeClassifier(), cart_params),
              ('RF', RandomForestClassifier(), rf_params),
              ("AdaBoost", AdaBoostClassifier(), adaboost_params),
              ('GBM', GradientBoostingClassifier(), gbm_params)]

    print("###################### Model Results 2######################")

    for name, classifier, params in models:
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X_train, y_train)
        final_model = classifier.set_params(**gs_best.best_params_).fit(X_train, y_train)
        y_train_pred = final_model.predict(X_train)
        y_test_pred = final_model.predict(X_test)

        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        acc_all = accuracy_score(y, final_model.predict(X))
        r2 = final_model.score(X, y)
        r2_train = final_model.score(X_train, y_train)
        r2_test = final_model.score(X_test, y_test)
        cv_train = cross_val_score(final_model, X_train, y_train, cv=cv).mean()
        cv_test = cross_val_score(final_model, X_test, y_test, cv=cv).mean()
        cv_all = cross_val_score(final_model, X, y, cv=cv).mean()
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

        results_dict['Model_Names'].append(name)
        results_dict['ACC_Train'].append(acc_train)
        results_dict['ACC_Test'].append(acc_test)
        results_dict['ACC_All'].append(acc_all)
        results_dict['R2'].append(r2)
        results_dict['R2_Train'].append(r2_train)
        results_dict['R2_Test'].append(r2_test)
        results_dict['CV_Train'].append(cv_train)
        results_dict['CV_Test'].append(cv_test)
        results_dict['CV_All'].append(cv_all)
        results_dict['Accuracy'].append(cv_results['test_accuracy'].mean())
        results_dict['Precision'].append(cv_results['test_precision'].mean())
        results_dict['Recall'].append(cv_results['test_recall'].mean())
        results_dict['F1-Score'].append(cv_results['test_f1'].mean())
        results_dict['Roc_Auc'].append(cv_results['test_roc_auc'].mean())
        results_dict['Best_Params'].append(gs_best.best_params_)

        with open(f'{name}_model.pkl', 'wb') as file:
            pickle.dump(classifier, file)

    model_tuned_results = pd.DataFrame(results_dict).set_index("Model_Names")
    model_tuned_results = model_tuned_results.sort_values(by="Accuracy", ascending=False)
    print(model_tuned_results)

    if plot:
        plt.figure(figsize=(15, 12))
        sns.barplot(x='Accuracy', y=model_tuned_results.index, data=model_tuned_results, color="r")
        plt.xlabel('Accuracy Values')
        plt.ylabel('Model Names')
        plt.title('Accuracy for All Models')
        plt.show()

    if save_results:
        model_tuned_results.to_csv("model_results.csv")

    return model_tuned_results


model_tuned_results = Create_and_Train_Classification_Model_Tuning(df, "Outcome", plot=True, save_results=True)



