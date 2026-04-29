from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import tkinter as tk      


# ==============================================================
#  BASIC PYTHON & SYSTEM UTILITIES
# ==============================================================
import os
import json
import base64
import pickle
import warnings
import base64
import requests
warnings.filterwarnings('ignore')

# ==============================================================
#  NUMERICAL & DATA PROCESSING LIBRARIES
# ==============================================================
import numpy as np
import pandas as pd

# ==============================================================
#  IMAGE PROCESSING & VISUALIZATION
# ==============================================================
import cv2                           # OpenCV for image operations
import matplotlib.pyplot as plt       # Plotting / visualization
from PIL import Image, ImageDraw, ImageFont  # Image drawing utilities
from PIL import Image, ImageTk, ImageEnhance
# ==============================================================
#  MACHINE LEARNING (SKLEARN)
# ==============================================================
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
)

from ngboost.distns import k_categorical
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.utils import resample             
import joblib   # Saving & loading models
import seaborn as sns
import re

# Classifiers
from sklearn.linear_model import (
    LogisticRegression, Perceptron
)
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB
)
from sklearn.neighbors import (
    KNeighborsClassifier, RadiusNeighborsClassifier,
    NearestCentroid
)
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import pairwise_distances
from scipy.stats import mode


# ==============================================================
#  TRANSFORMER (TIMM + TORCH)
# ==============================================================
import torch
import timm
from torchvision import transforms

from keras.models import Sequential
from tensorflow.keras.models import model_from_json
from keras.layers import (
    Conv2D, Convolution2D, MaxPooling2D,
    Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.applications import (
    DenseNet121, Xception
)
from tensorflow.keras.applications.densenet import preprocess_input
from skimage import io, transform
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==============================================================
#  Database
# ==============================================================
import hashlib
from tinydb import TinyDB, Query



global filename
global X, Y
global model
global categories,model_folder


model_folder = "model"
os.makedirs(model_folder, exist_ok=True)   # create model folder if not exist

feature_file = os.path.join(model_folder, "final_ViT_features.npz")
base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')

accuracy = []
precision = []
recall = []
fscore = []

def uploadDataset():
    global filename,categories,categories,path
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")
   
def DenseNet121_feature_extraction():
    global X, Y, base_model,categories,filename
    text.delete('1.0', END)

    model_data_path = "model/X.npy"
    model_label_path_GI = "model/Y.npy"

    if os.path.exists(model_data_path) and os.path.exists(model_label_path_GI):
        X = np.load(model_data_path)
        Y = np.load(model_label_path_GI)
    else:
 
        X = []
        Y = []
        data_folder=filename
        for class_label, class_name in enumerate(categories):
            class_folder = os.path.join(data_folder, class_name)
            for img_file in os.listdir(class_folder):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                    img_path = os.path.join(class_folder, img_file)
                    print(img_path)
                    img = image.load_img(img_path, target_size=(128,128))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = base_model.predict(x)
                    features = np.squeeze(features)  # Flatten the features
                    X.append(features)
                    Y.append(class_label)
        # Convert lists to NumPy arrays
        X = np.array(X)
        Y = np.array(Y)

        # Save processed images and labels
        np.save(model_data_path, X)
        np.save(model_label_path_GI, Y)
            
    text.insert(END, "Image Preprocessing Completed\n")
    text.insert(END, "DenseNet121 Feature Extraction completed\n")
    text.insert(END, f"Feature Dimension: {X.shape}\n")


def Model_Perceptron():
    global X, Y, base_model,categories,filename

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    text.delete('1.0', END)
    model_filename = os.path.join(model_folder, "Perceptron_model.pkl")
    
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = Perceptron(
        max_iter=1,         
        eta0=0.0001,        
        random_state=42,
        tol=None,          
        shuffle=False  )
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename,compress=5)
    
    Y_pred   = Model1.predict(X_test)
    y_scores = Model1.decision_function(X_test)
    Calculate_Metrics("DenseNet121-Perceptron", Y_pred, y_test, y_scores)



def Model_NearestCentroid():
    global X, Y, base_model,categories,filename

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    text.delete('1.0', END)
    
    model_filename = os.path.join(model_folder, "NearestCentroid_model.pkl")
    
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = NearestCentroid()
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)
    
    Y_pred = Model1.predict(X_test)
    distances = pairwise_distances(X_test, Model1.centroids_)
    y_scores = -distances[np.arange(len(X_test)), Y_pred]
    Calculate_Metrics("DenseNet121-NC", Y_pred, y_test, y_scores)


def Model_KNN_Radius_Combined():
    global X, Y, base_model,categories,filename

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "KNN_RadiusCombined_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        radius = RadiusNeighborsClassifier(radius=1.0, outlier_label='most_frequent')

        Model1 = VotingClassifier(estimators=[
            ('knn', knn),
            ('radius', radius)
        ], voting='hard')

        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)

    Y_pred = Model1.predict(X_test)
    Calculate_Metrics("DenseNet121-KNN-RNC", Y_pred, y_test, Y_pred)


def Initialize_ViT():
    global model, preprocess, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        "vit_base_patch16_224",   # ViT model
        pretrained=True,
        num_classes=0             # outputs feature embedding (not logits)
    )
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])



# ---------------------------
# VIT FEATURE EXTRACTION
# ---------------------------
def ViT_Feature_Extraction():
    global filename, categories, X, Y
    global model, preprocess, device

    Initialize_ViT()
    text.delete('1.0', END)
    text.insert(END, "Starting ViT (Vision Transformer) Feature Extraction...\n")

    # --------------------------------------------------
    # LOAD SAVED FEATURES IF EXISTS
    # --------------------------------------------------
    if os.path.exists(feature_file):
        text.insert(END, "Saved feature file found. Loading existing features...\n")
        data = np.load(feature_file, allow_pickle=True)
        X = data["features"]
        Y = data["labels"]
        text.insert(END, "Loaded saved features successfully.\n")
        return X, Y

    text.insert(END, "No saved file found. Extracting new ViT features...\n")

    X_features = []
    Y_labels = []

    # Loop through all classes
    for class_name in categories:
        class_folder = os.path.join(filename, class_name)

        text.insert(END, f"Processing Class: {class_name}\n")

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            print(img_path)

            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            # Load & preprocess image
            img = Image.open(img_path).convert("RGB")
            img = preprocess(img)
            img = img.unsqueeze(0).to(device)

            # Extract ViT Features
            with torch.no_grad():
                feat = model(img)   # 768-dimensional features

            feat_np = feat.cpu().numpy().flatten()

            X_features.append(feat_np)
            Y_labels.append(class_name)

    # Convert lists to arrays
    X_features = np.array(X_features)
    Y_labels = np.array(Y_labels)

    text.insert(END, "Feature extraction completed.\n")
    text.insert(END, f"Total samples processed: {len(Y_labels)}\n")

    # Save features
    np.savez(feature_file, features=X_features, labels=Y_labels)
    text.insert(END, f"Saved features to: {feature_file}\n")

    return X_features, Y_labels
    
def Train_Test_split():
    global X, Y, x_train, x_test, y_train, y_test

    text.delete('1.0', END)
    text.insert(END, "Starting Train-Test Split with Resampling...\n")

    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]
    

    x_train, x_test, y_train, y_test = train_test_split(
        X, 
        Y,
        test_size=0.20,
        random_state=42,
        stratify = Y# maintain class balance
    )

    # Print results in GUI
    text.insert(END, f"Training dataset size: {x_train.shape}\n")
    text.insert(END, f"Testing dataset size : {x_test.shape}\n")

    text.insert(END, "Train-Test splitting complete.\n")

    return x_train, x_test, y_train, y_test

def plot_confusion_matrix_numeric(y_true, y_pred, class_names, title="Confusion Matrix"):

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    # CASE 1: labels are already numeric
    if np.issubdtype(y_true.dtype, np.integer):
        y_true_num = y_true
        y_pred_num = y_pred
        num_classes = len(class_names)

    # CASE 2: labels are strings → map them
    else:
        class_to_num = {str(cls): i for i, cls in enumerate(class_names)}

        def safe_map(label):
            label = str(label)
            if label not in class_to_num:
                raise ValueError(f"Unknown label '{label}' not in class_names")
            return class_to_num[label]

        y_true_num = np.array([safe_map(c) for c in y_true])
        y_pred_num = np.array([safe_map(c) for c in y_pred])
        num_classes = len(class_names)

    # Confusion matrix
    cm = confusion_matrix(y_true_num, y_pred_num, labels=range(num_classes))

    # -------- Figure Layout --------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4])

    # -------- LEFT LEGEND --------
    ax_legend = fig.add_subplot(gs[0, 0])
    ax_legend.axis("off")

    legend_text = "CLASS NUMBER LEGEND:\n\n"
    for i, cls in enumerate(class_names):
        legend_text += f"{i} → {cls}\n"

    ax_legend.text(0, 1, legend_text,
                   fontsize=12,
                   verticalalignment="top",
                   fontfamily="monospace")

    # -------- RIGHT HEATMAP --------
    ax_cm = fig.add_subplot(gs[0, 1])

    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Purples",
                cbar=False,
                xticklabels=np.arange(num_classes),
                yticklabels=np.arange(num_classes),
                ax=ax_cm)

    ax_cm.set_title(title, fontsize=16)
    ax_cm.set_xlabel("Predicted Class (Number)", fontsize=14)
    ax_cm.set_ylabel("True Class (Number)", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_')}_CM.png")
    plt.show()


def Calculate_Metrics( algorithm, predict, y_test,y_score):
    global categories
    if not os.path.exists('results'):
        os.makedirs('results')
        
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plot_confusion_matrix_numeric(y_test, predict, categories, title=f"{algorithm} CM")

    if y_score is not None:

        # Convert output labels to numeric if needed
        class_indices = {c: i for i, c in enumerate(categories)}
        y_test_num = np.array([class_indices[y] for y in y_test])

        # One-vs-Rest binarization
        y_test_bin = label_binarize(y_test_num, classes=range(len(categories)))

        # Number of classes
        n_classes = y_test_bin.shape[1]

        fpr = {}
        tpr = {}
        roc_auc = {}

        # ---------------------------------------------
        # COMPUTE ROC FOR EACH CLASS
        # ---------------------------------------------
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # ---------------------------------------------
        # MICRO-AVERAGE ROC (global)
        # ---------------------------------------------
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # ---------------------------------------------
        # PLOT ROC CURVES
        # ---------------------------------------------
        plt.figure(figsize=(10, 8))

        # Plot each class ROC
        for i in range(n_classes):
            plt.plot(
                fpr[i], tpr[i],
                label=f'{categories[i]} (AUC = {roc_auc[i]:.2f})',
                linewidth=2
            )

        # Plot micro-average
        plt.plot(
            fpr["micro"], tpr["micro"],
            linestyle='--', linewidth=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})'
        )

        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.title(f"{algorithm} ROC Curves (One-vs-Rest)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
        plt.show()

        text.insert(END, "\nAUC Scores (per class):\n")
        for i in range(n_classes):
            text.insert(END, f"{categories[i]} : {roc_auc[i]:.4f}\n")

        text.insert(END, f"\nMicro-average AUC: {roc_auc['micro']:.4f}\n\n")


    
def Proposed_Final():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "Final_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading saved  model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training new model...\n")
        mlmodel = Perceptron()
   
        # Train model
        mlmodel.fit(x_train, y_train)
        # Save model
        joblib.dump(mlmodel, model_filename, compress=5)
        text.insert(END, f"Model saved at: {model_filename}\n")

    # Predict
    y_pred = mlmodel.predict(x_test)
    y_scores = mlmodel.decision_function(x_test)
    Calculate_Metrics("ViT-Perceptron", y_pred, y_test,y_scores)


def predict_original():
    global model_folder, categories,preprocess,model

    text.delete('1.0', END)
    text.insert(END, "Starting Prediction on Test Image...\n")

    test_image_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )




    img = Image.open(test_image_path).convert("RGB")
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    text.insert(END, "Extracting features using ViT...\n")
    with torch.no_grad():
        feature = model(img_tensor)

    feature_np = feature.cpu().numpy().flatten().reshape(1, -1)
    
    model_filename = os.path.join(model_folder, "SRC_model.pkl")

    final_model = joblib.load(model_filename)
    prediction = final_model.predict(feature_np)[0]
    text.insert(END, f"Prediction Completed: {prediction}\n")

    # --------------------------------------------------
    # DISPLAY RESULT ON IMAGE
    # --------------------------------------------------
    img_cv = cv2.imread(test_image_path)
    img_display = cv2.resize(img_cv, (500, 500))

    cv2.putText(img_display, f'Classified as: {prediction}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Prediction", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    text.insert(END, "Prediction displayed successfully.\n")

def predict_transformer(test_image_path):
    global model_folder, categories,preprocess,model


    img = Image.open(test_image_path).convert("RGB")
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    text.insert(END, "Extracting features using ViT...\n")
    with torch.no_grad():
        feature = model(img_tensor)

    feature_np = feature.cpu().numpy().flatten().reshape(1, -1)
    
    model_filename = os.path.join(model_folder, "Final_model.pkl")

    final_model = joblib.load(model_filename)
    prediction = final_model.predict(feature_np)[0]
    text.insert(END, f"Prediction Completed: {prediction}\n")
    return prediction

def draw_classification_text(
    img,
    prediction,
    x=10,
    y=40,
    max_scale=1.0,
    min_scale=0.4,
    line_gap=40,
    color=(255, 0, 0),
    thickness=2
):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # -------- Line 1 (Fixed text) --------
    cv2.putText(
        img,
        "Predicted:",
        (x, y),
        font,
        max_scale,
        color,
        thickness
    )

    # -------- Line 2 (Auto-adjust text) --------
    text = str(prediction)
    font_scale = max_scale

    (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)

    while text_w > w - 20 and font_scale > min_scale:
        font_scale -= 0.05
        (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.putText(
        img,
        text,
        (x, y + line_gap),
        font,
        font_scale,
        color,
        thickness
    )

    return img



def predict():
    global model_folder, categories, transform, model
    Initialize_ViT()
    

    text.delete('1.0', END)
    text.insert(END, "Starting Prediction on Test Image...\n")

    # --------------------------------------------------
    # STEP 1: SELECT TEST IMAGE
    # --------------------------------------------------
    test_image_path = filedialog.askopenfilename( initialdir="TestImages", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not test_image_path:
        text.insert(END, "No image selected.\n")
        return

    # --------------------------------------------------
    # STEP 2: RUN XAI Plant Leaf DETECTION
    # --------------------------------------------------
    result = analyze_plant_leaf_image(test_image_path)

    # Convert JSON string to dictionary
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            text.insert(END, "Error: XAI returned invalid JSON.\n")
            return

    print("XAI Result:", result)

    # --------------------------------------------------
    # CASE C: XAI FAILED → SHOW ONLY ORIGINAL + TRANSFORMER RESULT
    # --------------------------------------------------
    if "error" in result:
        text.insert(END, "⚠ XAI Failed: Showing Only Transformer Prediction.\n")

        prediction = predict_transformer(test_image_path)
        text.insert(END, f"Classifier Prediction: {prediction}\n")

        plt.figure(figsize=(14, 6))

        # 121 → Original Image
        plt.subplot(121)
        orig = cv2.imread(test_image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original Image")
        plt.axis("off")

        # 122 → Transformer Prediction
        plt.subplot(122)
        img_cls = cv2.imread(test_image_path)
        img_cls = cv2.cvtColor(img_cls, cv2.COLOR_BGR2RGB)
        img_cls = cv2.resize(img_cls, (500, 500))
        img_cls = draw_classification_text(img_cls, prediction)
        plt.imshow(img_cls)
        plt.title("Transformer Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

    # --------------------------------------------------
    # CASE A: Plant Leaf 
    # --------------------------------------------------
    if result.get("is_plant_leaf") == True:
        text.insert(END, "Plant Leaf detected. Running model classifier...\n")

        # CLASSIFIER PREDICTION
        prediction = predict_transformer(test_image_path)
        text.insert(END, f"Classifier Prediction: {prediction}\n")

        # ---------- CREATE Plant Leaf ANALYSIS WHITE PANEL ----------
        width, height = 800, 600
        result_img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(result_img)

        try:
            font_big = ImageFont.truetype("arial.ttf", 24)
        except:
            font_big = ImageFont.load_default()

        y = 30
        draw.text((30, y), "Plant Leaf Analysis Results", fill="black", font=font_big)
        y += 50

        for key, value in result.items():
            draw.text((30, y), f"{key}: {value}", fill="black", font=font_big)
            y += 40

        result_np = np.array(result_img)

        # ---------- DISPLAY 3 SUBPLOTS ----------
        plt.figure(figsize=(18, 6))

        # 131 → Original Plant Leaf
        plt.subplot(131)
        orig = cv2.imread(test_image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original Plant Leaf")
        plt.axis("off")

        # 132 → XAI Plant Leaf Result Panel
        plt.subplot(132)
        plt.imshow(result_np)
        plt.title("XAI Plant Leaf Result")
        plt.axis("off")

        # 133 → Classifier Result
        plt.subplot(133)
        img_cls = cv2.imread(test_image_path)
        img_cls = cv2.cvtColor(img_cls, cv2.COLOR_BGR2RGB)
        img_cls = cv2.resize(img_cls, (500, 500))
        img_cls = draw_classification_text(img_cls, prediction)
        plt.imshow(img_cls)

        plt.title("Model Classification Result")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

    # --------------------------------------------------
    # CASE B: NOT AN Plant Leaf → NORMAL XAI OUTPUT
    # --------------------------------------------------
    else:
        text.insert(END, f"Not an Plant Leaf. Detected Image Type = {result.get('image_type')}\n")

        width, height = 800, 600
        result_img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(result_img)

        try:
            font_big = ImageFont.truetype("arial.ttf", 24)
        except:
            font_big = ImageFont.load_default()

        y = 30
        draw.text((30, y), "Image Analysis Result", fill="black", font=font_big)
        y += 50

        for key, value in result.items():
            draw.text((30, y), f"{key}: {value}", fill="black", font=font_big)
            y += 40

        result_np = np.array(result_img)

        # ---------- DISPLAY 2 SUBPLOTS ----------
        plt.figure(figsize=(14, 6))

        # 121 → Original Image
        plt.subplot(121)
        orig = cv2.imread(test_image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original Image")
        plt.axis("off")

        # 122 → XAI Result
        plt.subplot(122)
        plt.imshow(result_np)
        plt.title("Detected Image Type")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

 
    
def analyze_plant_leaf_image(image_path):
    api_key = "AIzaSyA5CPxZsUQFJpdSlUwmzw0We_1U9MfulA0"  # your API key
    try:
        # Check if the image file exists
        if not os.path.exists(image_path):
            return json.dumps({"error": "File not found"})
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        if not api_key:
            return json.dumps({"error": "API key is missing"})

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key}"
        )

        # ----------------------------------
        # MAIN REQUEST PAYLOAD
        # ----------------------------------
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
                        {
                            "text": (
                                "Analyze the uploaded image.\n\n"
                                "TASK 1 — IMAGE TYPE DETECTION:\n"
                                "Determine whether the image is a Plant Leaf scene.\n\n"
                                "If NOT a Plant Leaf, return what the image contains in ONE WORD, for example:\n"
                                "  'Fish', 'Turtle', 'Human', 'Food', 'Animal', 'Object', etc.\n\n"
                                "TASK 2 — IF Plant Leaf:\n"
                                "If the image IS a Plant Leaf image, extract the following:\n"
                                "  - plant_present: true/false\n"
                                "  - plant_type: (approximate one word)\n"
                                "  - health_status: (one word)\n"
                                "  - visibility: (one word: 'Low', 'Medium', 'High')\n"
                                "  - dominant_color: (blue/green/brown/etc.)\n\n"
                                "RETURN STRICT JSON ONLY IN THIS FORMAT:\n"
                                "{\n"
                                "  \"is_plant_leaf\": true/false,\n"
                                "  \"image_type\": \"Plant Leaf\" or \"Fish\" or \"Human\" or \"Object\" etc,\n"
                                "  \"plant_present\": true/false,\n"
                                "  \"plant_type\": \"Boulder\" or null,\n"
                                "  \"health_status\": \"Healthy\" or null,\n"
                                "  \"visibility\": \"High\" or null,\n"
                                "  \"dominant_color\": \"Blue\" or null\n"
                                "}\n\n"
                                "Return ONLY JSON. No explanation."
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4000}
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        response_json = response.json()

        if "candidates" not in response_json or not response_json["candidates"]:
            return json.dumps({"error": "No candidates in API response"})

        candidate = response_json["candidates"][0]

        if (
            "content" in candidate
            and "parts" in candidate["content"]
            and len(candidate["content"]["parts"]) > 0
        ):
            text_response = candidate["content"]["parts"][0].get("text", "")
        else:
            return json.dumps({"error": "Unexpected API response format"})

        # Extract JSON from response
        json_match = re.findall(r"\{.*\}", text_response, re.DOTALL)
        if not json_match:
            return json.dumps({"error": "No JSON found in API response"})

        try:
            result_json = json.loads(json_match[0])
        except json.JSONDecodeError:
            return json.dumps({"error": "JSON decode failed"})

        return json.dumps(result_json, indent=4)

    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})



def close():
    main.destroy()

# =============================
# ⚙️ Database Setup (TinyDB)
# =============================

DB_PATH = "users_auth_tinydb.json"
db = TinyDB(DB_PATH)
users_table = db.table("users")

# =============================
# 🔒 Security (Password Hashing)
# =============================

def hash_password(password: str) -> str:
    """Generate a secure SHA-256 hash for the given password."""
    return hashlib.sha256(password.encode()).hexdigest()


# =============================
# 💾 Database Helpers
# =============================

def user_exists(username: str) -> bool:
    """Check if a user already exists."""
    User = Query()
    return users_table.contains(User.username == username)

def user_get(username: str):
    """Retrieve a user record."""
    User = Query()
    return users_table.get(User.username == username)

def user_add(username: str, password_hash: str, role: str):
    """Add a new user record."""
    users_table.insert({"username": username, "password": password_hash, "role": role})


# =============================
# 📝 Signup Function
# =============================

def signup(role: str):
    def register_user():
        username = username_entry.get().strip()
        password = password_entry.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Please fill all fields!")
            return

        if user_exists(username):
            messagebox.showerror("Error", "User already exists!")
            return

        hashed_pw = hash_password(password)
        user_add(username, hashed_pw, role)
        messagebox.showinfo("Success", f"{role} Signup Successful!")
        signup_window.destroy()

    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)

    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)


# =============================
# 🔐 Login Function
# =============================

def login(role: str):
    def verify_user():
        username = username_entry.get().strip()
        password = password_entry.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Please fill all fields!")
            return

        user = user_get(username)
        if not user:
            messagebox.showerror("Error", "User not found!")
            return

        hashed_pw = hash_password(password)
        if user["password"] == hashed_pw and user["role"] == role:
            messagebox.showinfo("Success", f"{role} Login Successful!")
            login_window.destroy()
            if role == "Admin":
                show_admin_buttons()
            else:
                show_user_buttons()
        else:
            messagebox.showerror("Error", "Invalid credentials or role mismatch!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)


# =============================
# 🧭 Dashboard Logic
# =============================

def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

def show_admin_buttons():
    clear_buttons()
    ff = ('times', 12, 'bold')

    uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
    uploadButton.place(x=20,y=550)
    uploadButton.config(font=ff)

    processButton = Button(main, text="DenseNet121", command=DenseNet121_feature_extraction)
    processButton.place(x=250,y=550)
    processButton.config(font=ff)
    
    processButton = Button(main, text="DenseNet121-Perceptron", command=Model_Perceptron)
    processButton.place(x=450,y=550)
    processButton.config(font=ff)

    modelButton = Button(main, text="DenseNet121-NC", command=Model_NearestCentroid)
    modelButton.place(x=650,y=550)
    modelButton.config(font=ff)

    modelButton = Button(main, text="DenseNet121-KNN-RNC", command=Model_KNN_Radius_Combined)
    modelButton.place(x=20,y=600)
    modelButton.config(font=ff)

    modelButton = Button(main, text="ViT Features", command=ViT_Feature_Extraction)
    modelButton.place(x=250,y=600)
    modelButton.config(font=ff)

    modelButton = Button(main, text="Splitting", command=Train_Test_split)
    modelButton.place(x=450,y=600)
    modelButton.config(font=ff)

    modelButton = Button(main, text="ViT with Perceptron", command=Proposed_Final)
    modelButton.place(x=650,y=600)
    modelButton.config(font=ff)

def show_user_buttons():
    clear_buttons()
    font1 = ('times', 13, 'bold')
    ff = ('times', 12, 'bold')
    predictButton = Button(main, text="Prediction from Test Image", command=predict)
    predictButton.place(x=650,y=600)
    predictButton.config(font=ff)


main = Tk('Plant Leaf')
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
window_width = int(screen_width)
window_height = int(screen_height )
main.geometry(f"{window_width}x{window_height}")

bg_image = Image.open("background.png")  # replace with your path
bg_image = bg_image.resize((screen_width, screen_height))
bg_photo = ImageTk.PhotoImage(bg_image)

background_label = tk.Label(main, image=bg_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


font = ('times', 18, 'bold')
font1 = ('times', 12, 'bold')
title = Label(main, text='OPTIMIZING VISION TRANSFORMERS FOR PLANT DISEASE CLASSIFICATION')
title.config(bg='white', fg='Red')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(relx=0.5, y=5, anchor='n')  # centers at top

# Main Buttons
admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1)
admin_button.place(x=20, y=250)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1)
user_button.place(x=20, y=300)

tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1).place(x=20, y=150)
tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1).place(x=20, y=200)



graphButton = Button(main, text="Exit", command=close)
graphButton.place(x=20,y=500)
graphButton.config(font=font1)


text=Text(main,height=28,width=50)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=850,y=100)
text.config(font=font1)

main.config(bg = 'DarkOrchid3')
main.mainloop()


