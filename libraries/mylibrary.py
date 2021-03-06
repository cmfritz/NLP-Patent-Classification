# Module of custom functions
#################################################################
def tsv_to_df(url):
    '''
    Given a url for PatentsView data, function downloads and returns
    a dataframe.
    
    Parameters: 
    url: string for url to PatentsView data (list found here:
    https://patentsview.org/download/data-download-tables)
    
    Returns:
    df: dataframe with the data downloaded and unzipped
    
    '''
    from pandas import read_csv
    df = read_csv(url, compression='zip', sep='\t', dtype=str)
    return df
    
#################################################################   
def merge_and_clean(df1, df2):
    '''
    Given two dataframes for my data, rename columns and merge data
    
    Parameters: 
    df1: 
    df2: 
    
    Returns:
    df: dataframe with the data downloaded and unzipped
    
    '''
    df1 = df1.drop(['field_title'], axis=1)
    df1 = df1.drop_duplicates()
    groups = df1.groupby(by='patent_id').count()
    singles = groups[groups.sector_title == 1]
    singles = singles.reset_index()
    ok_pats = df1.merge(singles, on='patent_id')
    ok_pats = ok_pats.drop(['sector_title_y'], axis=1)
    ok_pats = ok_pats.rename(columns={'sector_title_x':'sector'})
    ok_pats.patent_id = ok_pats.patent_id.astype('int64')
    df_merged = ok_pats.merge(df2, how='inner', on='patent_id')
    return df_merged

#################################################################
def preprocess_text(text):
    '''
    Given a string, returns tokenized and stemmed list of words
    
    Parameters: 
    text: string to be tokenized and stemmed
    
    Returns:
    keywords: list of stemmed tokens
    
    '''
    # Import packages and modules
    import pandas as pd
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords

    # Tokenize words while ignoring punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    # Create stopwords
    stop_words = stopwords.words('english')
    custom_stops = ['claim', 'claims', 'method', 'comprising', 'comprises', 'including', 'includes', 'according']
    [stop_words.append(word) for word in custom_stops]

    # Lowercase and stem
    stemmer = PorterStemmer()
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item.lower()))
    
    # Remove stopwords
    keywords= [stem for stem in stemmed if stem not in stop_words]
    return stemmed
    
#################################################################
def train_and_predict(model, X_train, X_test, y_train):
    '''
    Given a model and train and test data, function fits the model and returns predicted values.
    
    Parameters: 
    model: object containing initialized model for training
    Xy: object containing train and test data for X and y sets
    
    Returns:
    Xy: updated Xy object with X & y prediction data (y_hat_train and y_hat_test)
    
    '''
    model.fit(X_train, y_train)
    
    # Make predictions for test data
    y_hat_train = model.predict(X_train)
    
    # Make predictions for test data
    y_hat_test = model.predict(X_test)
    
    return y_hat_train, y_hat_test

#################################################################   
def model_scores(model, model_name,
                 X_train, X_test, 
                 y_train, y_test):
    '''
    Takes untrained model and test train data objects and generates scores and confusion matrix.
    
    Parameters: 
    model: object containing initialized model for training
    Xy: object containing train and test data for X and y sets
    
    Returns:
    printed text and visualizations
    '''

    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.metrics import plot_confusion_matrix
    from pandas import DataFrame
    from matplotlib.pyplot import savefig, title

    y_hat_train, y_hat_test = train_and_predict(model, X_train, X_test, y_train)

    scores = {}

    # Calculate accuracy for test data 
    scores['train_accuracy'] = accuracy_score(y_train, y_hat_train)

    # Calculate accuracy for test data
    scores['test_accuracy'] = accuracy_score(y_test, y_hat_test)
    
    no_classes = len(set(y_train))

    if no_classes < 3:
        # Calculate recall for train data
        scores['train_recall'] = recall_score(y_train, y_hat_train, pos_label='EE')
    
        # Calculate recall for test data
        scores['test_recall'] = recall_score(y_test, y_hat_test, pos_label='EE')
    
        # Calculate F1 for train data
        scores['train_f1'] = f1_score(y_train, y_hat_train, pos_label='EE')
    
        # Calculate F1 for test data
        scores['test_f1'] = f1_score(y_test, y_hat_test, pos_label='EE')
    
    # Print metrics
    print(scores)

    # Plot confusion matrix for train data
    plot_confusion_matrix(model, X=X_train, y_true=y_train,
                          values_format = 'd', xticks_rotation='vertical')
    title(f'{model_name} - Train Data Confusion Matrix')
    savefig(f'images/cm_{model_name}_Train_{no_classes}class.png', bbox_inches='tight', dpi=300)
 
    # Plot confusion matrix for test data
    plot_confusion_matrix(model, X=X_test, y_true=y_test,
                          values_format = 'd', xticks_rotation='vertical')
    title(f'{model_name} - Test Data Confusion Matrix')
    savefig(f'images/cm_{model_name}_Test_{no_classes}class.png', bbox_inches='tight', dpi=300)
    
    
    return DataFrame(scores, index=[model_name])

#################################################################   
def model_scores_pretrained(model, model_name,
                 X_train, X_test, 
                 y_train, y_test,
                 disp_conf_matrix=True):
    '''
    Takes already trained model and test train data objects and generates scores and confusion matrix.
    
    Parameters: 
    model: object containing trained model
    Xy: object containing train and test data for X and y sets
    
    Returns:
    printed text and visualizations
    '''

    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.metrics import plot_confusion_matrix
    from pandas import DataFrame
    from matplotlib.pyplot import savefig, title
    
    # Make predictions for test data
    y_hat_train = model.predict(X_train)
    
    # Make predictions for test data
    y_hat_test = model.predict(X_test)
    
    scores = {}

    # Caclulate accuracy for test data= 
    scores['train_accuracy'] = accuracy_score(y_train, y_hat_train)

    # Caclulate accuracy for test data
    scores['test_accuracy'] = accuracy_score(y_test, y_hat_test)
    
    no_classes = len(set(y_train))

    if no_classes < 3:
        # Caclulate recall for train data
        scores['train_recall'] = recall_score(y_train, y_hat_train, pos_label='EE')
    
        # Calculate recall for test data
        scores['test_recall'] = recall_score(y_test, y_hat_test, pos_label='EE')
    
        # Caclulate F1 for train data
        scores['train_f1'] = f1_score(y_train, y_hat_train, pos_label='EE')
    
        # Calculate F1 for test data
        scores['test_f1'] = f1_score(y_test, y_hat_test, pos_label='EE')

    # Print metrics
    print(scores)
    if disp_conf_matrix is True:
        # Plot confusion matrix for train data
        plot_confusion_matrix(model, X=X_train, y_true=y_train,
                              values_format = 'd', xticks_rotation='vertical')
        title(f'{model_name} - Train Data Confusion Matrix')
        savefig(f'images/cm_{model_name}_Train_{no_classes}class.png', bbox_inches='tight', dpi=300)
     
        # Plot confusion matrix for test data
        plot_confusion_matrix(model, X=X_test, y_true=y_test,
                              values_format = 'd', xticks_rotation='vertical')
        title(f'{model_name} - Test Data Confusion Matrix')
        savefig(f'images/cm_{model_name}_Test_{no_classes}class.png', bbox_inches='tight', dpi=300)
    
    return DataFrame(scores, index=[model_name])

#################################################################  
def run_grid_search(params, model, X_train, y_train):
    '''
    Takes in parameters, model, and test train data, runs grid search, and 
    returns updated model and test train data.
    
    Parameters: 
    params: dict with parameters for grid search
    model: object containing initialized model for training
    Xy: object containing train and test data for X and y sets
    
    Returns:
    gs: object containing updated model
    Xy: object with test train data
    '''
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(model, params, scoring='accuracy', cv=None, n_jobs=1)
    gs = gs.fit(X_train, y_train)
    best_parameters = gs.best_params_

    print('Grid Search found the following optimal parameters: ')
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))
    
    return gs