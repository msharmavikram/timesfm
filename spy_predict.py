import timesfm
from huggingface_hub import login
import numpy as np

login("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")

tfm = timesfm.TimesFm(
    context_len=32,
    horizon_len=16,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="cpu",
)

tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")


forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
print(forecast_input)
frequency_input = [0, 1, 2]
print(frequency_input)

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

print(point_forecast)
print(experimental_quantile_forecast)


