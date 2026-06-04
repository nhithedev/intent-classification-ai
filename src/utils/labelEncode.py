import numpy as np # type: ignore
# ========================================================
# 2. COMPONENT: LABEL ENCODER
# ========================================================
class LabelEncoder:
    def __init__(self):
        self.classes = []
        self.label_to_index = {}
        self.index_to_label = {}

    def fit_transform(self, labels):
        self.classes = sorted(list(set(labels)))
        self.label_to_index = {label: idx for idx, label in enumerate(self.classes)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.classes)}
        
        return np.array([self.label_to_index[label] for label in labels])
    
    def inverse_transform(self, indices):
        return [self.index_to_label[idx] for idx in indices]
