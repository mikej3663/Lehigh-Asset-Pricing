DEV = True  # Toggle for small dataset loading
WINDOW_SIZE = 5  # Rolling average window
AVAILABLE_MODELS = ['pred_mlp_32', 'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge']

MODEL_NAME_MAPPING = {
    'pred_mlp_32': 'NN1',
    'pred_mlp_64_32': 'NN2',
    'pred_mlp_128_64_32': 'NN3',
    'pred_hgbr': 'HistGradientBoosting',
    'pred_Lasso': 'Lasso',
    'pred_Ridge': 'Ridge'
}