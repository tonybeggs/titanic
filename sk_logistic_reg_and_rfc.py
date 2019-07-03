
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:

pd.options.display.max_rows = 100
pd.options.mode.chained_assignment = None


# ## Import Data

# In[3]:

train_data = pd.read_csv('train.csv')
train_data.shape


# In[4]:

test_data = pd.read_csv('test.csv')
test_data.head()


# In[5]:

test_data.shape


# ## Explore Dataset
# Check nulls, class balance and look at some value counts for different columns.

# In[6]:

# Count nulls in each column.
[[col, train_data[col].isnull().sum()] for col in train_data.columns]


# Look at the class balance. Makes sense to balance classes for model input data.
# Maybe look at bagging techniques.

# In[7]:

train_data['Survived'].value_counts(dropna=False)


# In[8]:

train_data['Pclass'].value_counts(dropna=False)


# In[9]:

train_data['Sex'].value_counts(dropna=False)


# Notably, `Age` is missing a significant number of values. Maybe we can later
# impute values using averages across other fields.

# In[10]:

train_data['Age'].value_counts(dropna=False).head(10)


# In[11]:

train_data['Embarked'].value_counts(dropna=False)


# Looking at each of the ports passengers left from:
# - There are significantly different class sizes
# - There are different survival rates

# In[12]:

def embarkment_port_pivot(input_df):
    '''Create a pivot counting the survival/deaths of passengers embarking
    from each separate port.
    '''
    
    df = input_df.pivot_table(
        index=['Embarked'],
        columns=['Survived'],
        values=['PassengerId'],
        aggfunc=len
    )
    
    df.columns = df.columns.droplevel()

    return df

df = embarkment_port_pivot(train_data)
df
df[1] / (df[0] + df[1])


# In[13]:

train_data['SibSp'].value_counts(dropna=False)


# In[14]:

train_data['Embarked'].value_counts(dropna=False)


# In[15]:

train_data['Name'].head()


# ## Feature Engineering
# Select and preprocess some features before modelling.

# In[16]:

def featurize_column(input_df, input_col, index_col=None):
    '''Convert a column of text labels into a dataframe of binary
    columns in each unique label value.
    
    Args:
    input_df (DataFrame):
    index_col (str):
    input_col (str):
    
    Returns:
    feature_piv (DataFrame):
    
    Example:
    
    User | Job               User | Carpenter | Electrician   
    Joe  | Carpenter    -->  Joe  |    1      |     0
    Jane | Electrician       Jane |    0      |     1

    '''
                    
    feature_piv = input_df[[input_col]]
    feature_piv['count'] = 1
    
    feature_piv = feature_piv.pivot_table(
        index=feature_piv.index,
        columns=[input_col],
        values=['count']
    )
    
    feature_piv.columns = feature_piv.columns.droplevel()
    feature_piv.columns = [input_col + '_' + col for col in feature_piv.columns]
    
    feature_piv = feature_piv.fillna(0)
    
    return feature_piv

def replace_with_featurized_column(input_df, input_col, index_col=None):
    '''Replace a column of text labels in a dataframe with a set of binary
    columns in each unique label value.
    
    Args:
    input_df (DataFrame):
    index_col (str):
    input_col (str):
    
    Returns:
    output_df (DataFrame):

    '''
                    
    feature_piv = featurize_column(input_df, input_col, index_col)
    
    output_df = input_df.drop(input_col, axis=1)
    output_df = output_df.join(feature_piv, how='left')
    
    return output_df


# In[17]:

def select_features(input_df):
    '''Select features to be input and preprocessed for model.
    '''
    
    output_df = input_df.set_index('PassengerId')
    
    features = [
        'Survived',
        'Name',
        'Pclass',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Fare',
        'Embarked'
    ]
    
    output_df = output_df[features]
    
    return output_df


# In[18]:

def strip_punctuation(input_str):
    '''Remove all punctuation characters from string. Leaves
    whitespace characters as is.
    '''
    stripped = ''.join(c for c in input_str if c not in punctuation)
    return stripped

def get_unique_name_words(input_df):
    '''Find the set of unique words in all names across the dataset.
    Useful for exploring bag of words analysis.
    '''
    s = input_df['Name'].apply(strip_punctuation).str.split(' ')
    s = s.apply(pd.Series).values
    
    s = pd.Series(np.concatenate(s)).dropna()
    s = s.value_counts(dropna=False)
    
    return s


# In[19]:

def get_married_female_column(input_df):
    ''' Find if a female is married by checking if her husband's name is
    contained in the 'Name' column, which seems to be demarcated by round
    brackets.
    '''
    output_df = input_df.copy()
    is_woman = output_df['Sex'] == 'female'
    is_married = ((output_df['Name'].str.split(' \(').str.len() + 1) % 2).astype(bool)
    
    output_df['IsMarriedWoman'] = (is_woman & is_married).astype(int)
    
    output_df = output_df.drop('Name', axis=1)
    
    return output_df


# In[20]:

def get_num_words_in_name(input_df):
    '''Find the number of words/terms in a passenger's name.
    '''
    
    num_words = input_df['Name'].apply(strip_punctuation).str.split(' ').str.len()
    
    return num_words



# In[21]:

def build_model_df(input_df):
    '''Preprocess features and rescale the data.
    '''
    
    model_df = select_features(input_df)
    
    model_df['num_words_in_name'] = get_num_words_in_name(model_df)
    
    model_df = get_married_female_column(model_df)
    
    # Turn sex into single binary column.
    model_df['Sex'] = (model_df['Sex'] == 'female').astype(int)
    
    # Turn embarkment port column into binary columns.
    for col in ['Embarked']:
        model_df = replace_with_featurized_column(model_df, col)
      
    # Store columns before dataframe gets turned into array by scaler.
    model_cols = model_df.columns
        
    # Rescale the data.
    model_df = model_df.dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    model_df = pd.DataFrame(
        data=scaler.fit_transform(model_df),
        columns=model_cols
    )
        
    return model_df

model_df = build_model_df(train_data)


# In[22]:

model_df.shape


# In[23]:

model_df.head()


# Examine the feature correlations in the model data:

# In[24]:

model_df.corr()


# Separate train and test datasets.

# In[25]:

features = model_df[model_df.columns[1:]]
targets = model_df[model_df.columns[0]]

X_train, X_test, y_train, y_test = train_test_split(
    features, 
    targets, 
    test_size=0.30, 
    random_state=1
)


# ## Build Model

# ### Logistic Regression
# Since the target is binary let's go with the logisitic regression estimator. This appears to be performing
# better than an SGD classifier with these input features.
# 
# First GridSearch over `C` and `penalty` params to find optimal model.

# In[26]:

logistic_reg = LogisticRegression(
    max_iter=100, 
    tol=1e-3, 
    solver='liblinear'
)

pipe = Pipeline(
    steps=[('logistic', logistic_reg)]
)

# Set pipeline parameters and their ranges.
param_grid = {
    'logistic__C': np.arange(0.1, 1.1, 0.1),
    'logistic__penalty': ['l1', 'l2'],
}

search = GridSearchCV(pipe, param_grid, iid=True, cv=5)
search.fit(X_train, y_train)
"Best parameter (CV score=%0.3f):" % search.best_score_
search.best_params_


# Evaluate logistic regression model using optimal params.

# In[27]:

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

'Mean prediction accuracy: ' + str(best_model.score(X_test, y_test))
'Precision score: ' + str(precision_score(y_test, y_pred, average='binary'))
'Recall score: ' + str(recall_score(y_test, y_pred, average='binary'))


# ### Random Forest Classifier
# The performance of the logistic regression is not great. Let's look at a random forest
# classifier.

# In[28]:

rfc = RandomForestClassifier()

pipe = Pipeline(
    steps=[('rfc', rfc)]
)

# Set pipeline parameters and their ranges.
param_grid = {
    'rfc__bootstrap': [True, False],
    'rfc__n_estimators': np.arange(15, 46, 5),
    'rfc__criterion': ['gini', 'entropy'],
}

search = GridSearchCV(pipe, param_grid, iid=True, cv=5)
search.fit(X_train, y_train)
"Best parameter (CV score=%0.3f):" % search.best_score_
search.best_params_


best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

'Mean prediction accuracy: ' + str(best_model.score(X_test, y_test))
'Precision score: ' + str(precision_score(y_test, y_pred, average='binary'))
'Recall score: ' + str(recall_score(y_test, y_pred, average='binary'))


# Let's try upsampling the minority class (survivors).

# In[29]:


def upsample_minority(X_train, y_train):
    '''Artificially create more samples in the minority class using
    k-nearest neighbours.
    '''

    # Concatenate training data back together.
    X = pd.concat([X_train, y_train], axis=1)

    # Separate minority and majority classes.
    died = X[X['Survived'] == 0]
    survived = X[X['Survived'] == 1]

    # Upsample minority class.
    survived_upsampled = resample(
        survived,
        replace=True,
        n_samples=died.shape[0],
        random_state=2
    )

    # Replace minority data with upsampled data.
    upsampled = pd.concat([died, survived_upsampled])

    upsamp_y_train = upsampled['Survived']
    upsamp_X_train = upsampled.drop('Survived', axis=1)
    
    return upsamp_X_train, upsamp_y_train

upsamp_X_train, upsamp_y_train = upsample_minority(X_train, y_train)

# upsampled_model = LogisticRegression(solver='liblinear')
upsampled_model = RandomForestClassifier(n_estimators=20)
upsampled_model.fit(upsamp_X_train, upsamp_y_train)

upsampled_pred = upsampled_model.predict(X_test)

'Mean prediction accuracy: ' + str(upsampled_model.score(X_test, y_test))
'Precision score: ' + str(precision_score(y_test, upsampled_pred, average='binary'))
'Recall score: ' + str(recall_score(y_test, upsampled_pred, average='binary'))

