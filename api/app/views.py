from .models import PredictionResponse, PredictionRequest
from .utils import get_model, transform_to_dataframe

model = get_model()

def get_prediction(request: PredictionRequest) -> float:
    df = transform_to_dataframe(request)
    prediction = model.predict(df)[0]
    return max(0, prediction)