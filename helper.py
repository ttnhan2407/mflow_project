import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_water_light_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    # Tạo 2 feature: nước (%) và ánh sáng (%)
    water = np.random.uniform(0, 100, n_samples)
    light = np.random.uniform(0, 100, n_samples)
    
    # Tính label: trung bình >= 80 => sống(1), <80 => chết(0)
    avg = (water + light)/2
    label = (avg >= 80).astype(int)
    
    X = pd.DataFrame({'water': water, 'light': light})
    y = label
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
