from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def update_model(model: Pipeline) -> None:
    """Update a model.    
    Args:
        model (Pipeline): A scikit-learn Pipeline.
    """
    dump(model, 'model/model.pkl')
    
def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
    with open('report.txt','w') as report_file:
        report_file.write('# Model Pipeline Description\n')
        
        for key, value in model.named_steps.items():
            report_file.write(f'### {key}: {value.__repr__()}\n')
            
        report_file.write(f'### Train score: {train_score}\n')
        report_file.write(f'### Test score: {test_score}\n')
        report_file.write(f'### Validation score: {validation_score}\n')

def get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) -> None:
    """Generate a report with the performance of the model in the test set.
    Args:
        y_real (pd.Series): The real values of the target variable.
        y_pred (pd.Series): The predicted values of the target variable.
    """
    plt.figure(figsize=(10,10))
    sns.regplot(x=y_pred, y=y_real)
    plt.xlabel('Predicted worldwide gross')
    plt.ylabel('Real worldwide gross')
    plt.title('Behavior of model prediction') 
    plt.savefig('prediction_behavior.png')
