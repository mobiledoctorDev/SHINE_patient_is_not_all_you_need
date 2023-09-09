"""
Don't forget to run automl_dataset.py before this script
- Myeongchan
"""
import os
from datetime import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/mckim/Key/shine-mobiledodctor-xxx.json"

import pandas as pd
from google.cloud import aiplatform
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

config = None
PROJECT_NAME = 'shine-mobiledodctor'
REGION = 'asia-northeast3'


def get_model_dir(model_result):
    return os.path.join('./train_result', model_result.display_name)


def save_fig(model_result):
    model_dir = get_model_dir(model_result)
    mean_attr = model_result.to_dict()['modelExplanation']['meanAttributions']
    shapey_values = mean_attr[0]['featureAttributions']
    df_features = pd.DataFrame()
    for key, val in shapey_values['featureAttributions'].items():
        df_features.loc[key, 'attributions'] = val
    df_features.sort_values('attributions', ascending=False)
    df_features['feature'] = df_features.index
    print(df_features)

    csv_path = os.path.join(model_dir, 'feature_importance.csv')
    df_features.to_csv(csv_path, index=False)
    sns.barplot(data=df_features.head(10),
                x='attributions',
                y='feature'
    )
    fig_path = os.path.join(model_dir, 'feature_importance_top_10.png')
    plt.savefig(filename=fig_path)
    plt.clf()


def get_models():
    location = REGION
    api_endpoint = "asia-northeast3-aiplatform.googleapis.com"

    # The AI Platform services require regional API endpoints.
    client_options = {
        "api_endpoint": api_endpoint,
    }
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.ModelServiceClient(client_options=client_options)
    models = client.list_models()

    from google.cloud import aiplatform_v1
    aiplatform_v1.types.model_evaluation.ModelEvaluation



def get_model_evaluation_tabular_classification_sample(
        project: str,
        model_id: str,
        evaluation_id: str,
        location: str = "asia-northeast3",
        api_endpoint: str = "asia-northeast3-aiplatform.googleapis.com",
):
    """
    To obtain evaluation_id run the following commands where LOCATION
    is the region where the model is stored, PROJECT is the project ID,
    and MODEL_ID is the ID of your model.

    model_client = aiplatform.gapic.ModelServiceClient(
        client_options={
            'api_endpoint':'LOCATION-aiplatform.googleapis.com'
            }
        )
    evaluations = model_client.list_model_evaluations(parent='projects/PROJECT/locations/LOCATION/models/MODEL_ID')
    print("evaluations:", evaluations)
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.ModelServiceClient(client_options=client_options)
    name = client.model_evaluation_path(
        project=project, location=location, model=model_id, evaluation=evaluation_id
    )
    response = client.get_model_evaluation(name=name)
    # print("response:", response)
    metrics = response.metrics()
    print(metrics)


if __name__ == '__main__':
    model = get_model_evaluation_tabular_classification_sample()
    res = model.get_model_evaluation()
    save_fig(res)
