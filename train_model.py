import os
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


DATA_DIR = "Train"

x = []
y = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".png") and "_" in fname:
        try:
            # Extract digit after last underscore
            label = int(fname.split("_")[-1].split(".")[0])
            path = os.path.join(DATA_DIR, fname)
            img = Image.open(path)
            img_data = np.array(img).astype(np.float32)
            img_data = 16 - img_data
            # Visual debug: only once
            x.append(img_data.flatten())
            y.append(label)
        except Exception as e:
            print(f"skipped {fname}: {e}")
X = np.array(x)
Y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Train model
clf = svm.SVC(kernel='linear', C=100)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on images: {acc:.2f}")
joblib.dump(clf, 'model.pkl')
print("Saved model")