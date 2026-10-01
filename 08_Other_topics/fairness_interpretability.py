# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: develop_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandeshjung/Machine-Learning-Foundation/blob/main/08_Other_topics/fairness_interpretability.ipynb)

# %%
# Colab setup: install the shared helpers (mlf_utils) and any extra packages. Does nothing when run locally.
import sys
if "google.colab" in sys.modules:
    get_ipython().run_line_magic("pip", "install -q git+https://github.com/sandeshjung/Machine-Learning-Foundation.git lime shap fairlearn")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import HTML, display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import lime
import lime.lime_tabular
import shap

sns.set_theme(style="whitegrid")
print(f"PyTorch Version: {torch.__version__}")

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# # Responsible AI: Fairness & Interpretability
#
# <p>As ML models become more powerful and integrated into critical decision-making processes (e.g., loan applications, medical diagnosis, hiring), it's crucial to ensure:
#
# - Fairness: The models do not disproportionately harm or benefit certain groups, especially those defined by sensitive attributes (race, gender, etc.).
# - Interpretability/ Explainability: We can understand why a model makes a particular prediction or behaves in a certain way. This builds trust, allows for debugging, ensure accountability, and can lead to new insights.</p>

# %% [markdown]
# ### Fairness in Machine Learning
#
# - Multiple definitions of fairness (e.g., group vs. individual).
# - Bias can originate from data, algorithm, or human interpretation.
# - Common metrics: Demographic Parity, Equalized Odds, Equal Opportunity.
# - Mitigation: Pre-processing data, in-processing algorithms, post-processing predictions.

# %% [markdown]
# ### Interpretability and Explainability (XAI)
#
# <p>Goal: Understand how ML models make decisions.
#
# - Model-Agnostic methods: Can be applied to any model (treat model as black box).
# - Model-Specific methods: Rely on the internal structure of specific models (e.g., tree-based feature importance).
# - Local Explanations: Explain an individual prediction.
# - Global Explanations: Try to understand the model's overall behavior.</p>

# %%
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']

os.makedirs("./data", exist_ok=True)

file_path = './data/adult.data'
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# %%
try:
    adult_df = pd.read_csv(file_path, header=None, names=column_names,
                           na_values=' ?', skipinitialspace=True)
except FileNotFoundError:
    print("adult.data not found locally. Downloading...")
    import urllib.request
    urllib.request.urlretrieve(data_url, file_path)
    print("Download complete.")
    adult_df = pd.read_csv(file_path, header=None, names=column_names,
                           na_values=' ?', skipinitialspace=True)

# %%
adult_df.drop(columns=['fnlwgt'], inplace=True, errors='ignore') # Remove if fnlwgt was in placeholder

adult_df.dropna(inplace=True)
adult_df.drop_duplicates(inplace=True)

# Convert income to binary
adult_df['income_binary'] = adult_df['income'].apply(lambda x: 1 if x == '>50K' else 0)
adult_df.drop(columns=['income'], inplace=True)

# %%
print(f"Shape: {adult_df.shape}")

# %%
print(adult_df.head())

# %%
# identify categorical and numerical features
categorical_features = adult_df.select_dtypes(include=['object']).columns.tolist()
numerical_features = adult_df.select_dtypes(include=np.number).drop(columns=['income_binary']).columns.tolist()
feature_names_processed = []

# %%
categorical_features, numerical_features

# %%
# preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# %%
preprocessor = ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder = 'passthrough'
)

# %%
X_adult = adult_df.drop(columns=['income_binary'])
y_adult_series = adult_df['income_binary']

# %%
X_processed_np = preprocessor.fit_transform(X_adult)
X_processed_np

# %%
try:
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names_processed.extend(numerical_features)
    feature_names_processed.extend(ohe_feature_names)
except AttributeError: # Older sklearn might not have get_feature_names_out
    print("Warning: Could not get OHE feature names automatically. Using generic names.")
    num_ohe_features = X_processed_np.shape[1] - len(numerical_features)
    ohe_feature_names = [f"cat_feat_{i}" for i in range(num_ohe_features)]
    feature_names_processed.extend(numerical_features)
    feature_names_processed.extend(ohe_feature_names)

# %%
X_processed_tensor = torch.from_numpy(X_processed_np).float().to(device)
y_adult_tensor = torch.from_numpy(y_adult_series.values).float().unsqueeze(1).to(device)

# %%
X_train_adult, X_test_adult, y_train_adult, y_test_adult = train_test_split(
    X_processed_tensor, y_adult_tensor, test_size=0.2, random_state=42, stratify=y_adult_tensor.cpu()
)

# %%
train_dataset_adult = TensorDataset(X_train_adult, y_train_adult)
train_loader_adult = DataLoader(train_dataset_adult, batch_size=64, shuffle=True)

# %%
print(f"Processed X_train_adult shape: {X_train_adult.shape}")
print(f"Number of features after OHE: {len(feature_names_processed)}")
if len(feature_names_processed) == X_train_adult.shape[1]:
    print("Feature names match processed data dimension.")
else:
    print(f"WARNING: Mismatch in feature names ({len(feature_names_processed)}) and data dim ({X_train_adult.shape[1]})")
    feature_names_processed = [f"feat_{i}" for i in range(X_train_adult.shape[1])]

# %%
input_dim = X_train_adult.shape[1]
hidden_dim = 64
output_dim = 1      # binary classification


# %%
class AdultMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdultMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Sigmoid will be applied outside or by BCEWithLogitsLoss

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x) # Output logits
        return x


# %%
adult_model = AdultMLP(input_dim, hidden_dim, output_dim).to(device)

# %%
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(adult_model.parameters(), lr=0.001)

# %%
num_epochs = 20
for epoch in range(num_epochs):
    adult_model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader_adult:
        optimizer.zero_grad()
        outputs_logits = adult_model(batch_X)
        loss = criterion(outputs_logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader_adult):.4f}")

# %%
adult_model.eval()
with torch.no_grad():
    y_pred_logits_test = adult_model(X_test_adult)
    y_pred_proba_test = torch.sigmoid(y_pred_logits_test)
    y_pred_class_test = (y_pred_proba_test >= 0.5).float()
    test_accuracy = accuracy_score(y_test_adult.cpu().numpy(), y_pred_class_test.cpu().numpy())
    print(f"Trained MLP Test Accuracy on Adult Dataset: {test_accuracy:.4f}")


# %% [markdown]
# ## Measuring Fairness
#
# About 85% accuracy says nothing about *who* the errors fall on. Below we compute the group fairness metrics from the [README](README.md#3-group-fairness-metrics) for two sensitive attributes, **sex** and **race**, first from scratch and then checked against [`fairlearn`](https://fairlearn.org).
#
# For each group $a$:
# - **Selection rate** $P(\hat{Y}=1 \mid A=a)$: how often the model predicts ">50K"
# - **TPR** $P(\hat{Y}=1 \mid Y=1, A=a)$: of the people who really earn >50K, how many the model finds
# - **FPR** $P(\hat{Y}=1 \mid Y=0, A=a)$: of the people who don't, how many it wrongly flags
#
# The fairness metrics summarise the **largest gap between any two groups** (0 = perfectly equal):
#
# | Metric | Definition |
# |---|---|
# | Demographic parity difference | $\max_a SR_a - \min_a SR_a$ |
# | Disparate impact ratio | $\min_a SR_a \,/\, \max_a SR_a$ (below 0.8 is a common red flag) |
# | Equal opportunity difference | $\max_a TPR_a - \min_a TPR_a$ |
# | Equalized odds difference | $\max(\text{TPR gap}, \text{FPR gap})$ |

# %%
# Recover each test row's sensitive attributes from its one-hot columns (they are 0/1, not scaled)
def group_labels(prefix):
    cols = [i for i, name in enumerate(feature_names_processed) if name.startswith(prefix)]
    names = np.array([feature_names_processed[i][len(prefix):] for i in cols])
    return names[X_test_adult[:, cols].cpu().numpy().argmax(axis=1)]

sex_test = group_labels("sex_")
race_test = group_labels("race_")
y_true_fair = y_test_adult.cpu().numpy().ravel().astype(int)
y_pred_fair = y_pred_class_test.cpu().numpy().ravel().astype(int)
y_proba_fair = y_pred_proba_test.cpu().numpy().ravel()

pd.Series(sex_test).value_counts(), pd.Series(race_test).value_counts()


# %%
def group_rates(y_true, y_pred, groups):
    """Per-group base rate, selection rate, TPR, FPR and accuracy."""
    rows = {}
    for g in np.unique(groups):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        rows[g] = {"n": int(m.sum()),
                   "base_rate": yt.mean(),              # P(Y=1 | A=g): how common >50K really is
                   "selection_rate": yp.mean(),         # P(Y_hat=1 | A=g)
                   "TPR": yp[yt == 1].mean(),           # P(Y_hat=1 | Y=1, A=g)
                   "FPR": yp[yt == 0].mean(),           # P(Y_hat=1 | Y=0, A=g)
                   "accuracy": (yt == yp).mean()}
    return pd.DataFrame(rows).T

def fairness_metrics(y_true, y_pred, groups):
    r = group_rates(y_true, y_pred, groups)
    gap = lambda col: r[col].max() - r[col].min()
    return {"demographic_parity_difference": gap("selection_rate"),
            "disparate_impact_ratio": r["selection_rate"].min() / r["selection_rate"].max(),
            "equal_opportunity_difference": gap("TPR"),
            "equalized_odds_difference": max(gap("TPR"), gap("FPR"))}

for name, groups in [("sex", sex_test), ("race", race_test)]:
    print(f"\n=== {name} ===")
    print(group_rates(y_true_fair, y_pred_fair, groups).round(3).to_string())
    print({k: round(float(v), 3) for k, v in fairness_metrics(y_true_fair, y_pred_fair, groups).items()})

# %% [markdown]
# ### Verifying against fairlearn

# %%
from fairlearn.metrics import (demographic_parity_difference, demographic_parity_ratio,
                               equal_opportunity_difference, equalized_odds_difference)
from mlf_utils import check_close

for name, groups in [("sex", sex_test), ("race", race_test)]:
    ours = fairness_metrics(y_true_fair, y_pred_fair, groups)
    kw = dict(y_true=y_true_fair, y_pred=y_pred_fair, sensitive_features=groups)
    check_close(f"{name}: demographic parity difference", ours["demographic_parity_difference"], demographic_parity_difference(**kw))
    check_close(f"{name}: disparate impact ratio", ours["disparate_impact_ratio"], demographic_parity_ratio(**kw))
    check_close(f"{name}: equal opportunity difference", ours["equal_opportunity_difference"], equal_opportunity_difference(**kw))
    check_close(f"{name}: equalized odds difference", ours["equalized_odds_difference"], equalized_odds_difference(**kw))

# %%
rates_sex = group_rates(y_true_fair, y_pred_fair, sex_test)
ax = rates_sex[["base_rate", "selection_rate", "TPR", "FPR"]].T.plot.bar(figsize=(8, 4), rot=0)
ax.set_title("Model behaviour by sex (test set)")
ax.set_ylabel("Rate")
plt.show()


# %% [markdown]
# **Reading the results.** Men are selected far more often than women. Part of that gap reflects a genuine difference in base rates in this 1994 census data, which a perfectly accurate model would also reproduce. Demographic parity ignores that, which is why it can conflict with accuracy. The **TPR and FPR gaps** are about *errors*: qualified women are missed more often than qualified men (equal opportunity), and the model's false alarms are unevenly distributed (equalized odds). Note also that the model sees `sex` directly as an input feature, and dropping it wouldn't be enough anyway, because `relationship` (husband/wife) is a near-perfect proxy.
#
# For **race**, the smallest groups have only 60–70 test samples and a handful of positives, so their TPRs are very noisy. The large race gaps are driven mostly by those groups. Always check group sizes (and ideally confidence intervals) before drawing conclusions.
#
# ### Mitigation by post-processing: group-specific thresholds
#
# A simple way to close the equal-opportunity gap without retraining is to use a **separate decision threshold per group**, chosen so every group reaches the same TPR as the overall model. This usually costs a little accuracy. That is the fairness/accuracy trade-off made explicit.

# %%
def threshold_for_tpr(proba, y_true, target_tpr):
    """Largest threshold whose TPR (on the positives in `y_true`) is at least `target_tpr`."""
    pos_scores = np.sort(proba[y_true == 1])[::-1]
    k = int(np.ceil(target_tpr * len(pos_scores))) - 1
    return pos_scores[max(k, 0)]

target_tpr = y_pred_fair[y_true_fair == 1].mean()      # the overall TPR at threshold 0.5
thresholds = {g: threshold_for_tpr(y_proba_fair[sex_test == g], y_true_fair[sex_test == g], target_tpr)
              for g in np.unique(sex_test)}
y_pred_post = np.array([y_proba_fair[i] >= thresholds[g] for i, g in enumerate(sex_test)]).astype(int)

print("Per-group thresholds:", {str(g): round(float(t), 3) for g, t in thresholds.items()})
before = fairness_metrics(y_true_fair, y_pred_fair, sex_test)
after = fairness_metrics(y_true_fair, y_pred_post, sex_test)
print(pd.DataFrame({"threshold 0.5": before, "group thresholds": after}).round(3).to_string())
print(f"Accuracy: {np.mean(y_pred_fair == y_true_fair):.4f} -> {np.mean(y_pred_post == y_true_fair):.4f}")

# %% [markdown]
# ### Try it: per-group thresholds
# Move each group's threshold and watch the per-group rates and the overall accuracy. Can you equalise TPR? What happens to the selection rates and FPR while you do?
#
# *Interactive: run the notebook locally or in Colab to use the controls. GitHub only renders a static page.*

# %%
from ipywidgets import FloatSlider, interact

@interact(threshold_female=FloatSlider(value=0.5, min=0.05, max=0.95, step=0.01, continuous_update=False),
          threshold_male=FloatSlider(value=0.5, min=0.05, max=0.95, step=0.01, continuous_update=False))
def explore_group_thresholds(threshold_female, threshold_male):
    t = np.where(sex_test == "Female", threshold_female, threshold_male)
    y_pred_t = (y_proba_fair >= t).astype(int)
    rates = group_rates(y_true_fair, y_pred_t, sex_test)
    fm = fairness_metrics(y_true_fair, y_pred_t, sex_test)
    ax = rates[["selection_rate", "TPR", "FPR", "accuracy"]].T.plot.bar(figsize=(8, 4), rot=0, ylim=(0, 1))
    ax.set_title(f"Overall accuracy = {np.mean(y_pred_t == y_true_fair):.3f} | "
                 f"TPR gap = {fm['equal_opportunity_difference']:.3f} | FPR gap = "
                 f"{rates['FPR'].max() - rates['FPR'].min():.3f}", fontsize=10)
    plt.show()


# %% [markdown]
# ## LIME (Local Interpretable Model-agnostic Explanations)
#
# <p>LIME explains individual predictions by learning a simple, interpretable model (e.g., linear regression) locally around the prediction. It generates a neighborhood of perturbed samples around the instance to explain, gets predictions for these neighbors from the black-box model, and then fits an interpretable model to these neighbor predictions, weighted by proximity.</p>

# %%
# LIME works best with numpy arrays
X_train_adult_np = X_train_adult.cpu().numpy()
class_names_adult = ['<=50K', '>50K']

# %%
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_adult_np,
    feature_names=feature_names_processed,
    class_names=class_names_adult,
    mode='classification',
    discretize_continuous=True
)

# %%
explainer_lime

# %%
instance_idx_lime = 0
instance_to_explain_lime_tensor = X_test_adult[instance_idx_lime] # Processed features
instance_to_explain_lime_np = instance_to_explain_lime_tensor.cpu().numpy()
true_label_lime = y_test_adult[instance_idx_lime].item()


# %%
def pytorch_predict_proba_for_lime(numpy_data):
    adult_model.eval() 
    tensor_data = torch.from_numpy(numpy_data).float().to(device)
    with torch.no_grad():
        logits = adult_model(tensor_data)
        probas = torch.sigmoid(logits) # Prob for class 1
    # LIME expects probabilities for all classes, shape (n_samples, n_classes)
    # For binary: [P(class 0), P(class 1)]
    return torch.cat((1 - probas, probas), dim=1).cpu().numpy()


# %%
print(f"\nExplaining instance {instance_idx_lime} from test set with LIME.")
print(f"True label: {class_names_adult[int(true_label_lime)]}")
predicted_proba_instance_lime = pytorch_predict_proba_for_lime(instance_to_explain_lime_np.reshape(1, -1))
predicted_class_idx_lime = np.argmax(predicted_proba_instance_lime)
print(f"Model's predicted probability for instance: {predicted_proba_instance_lime[0]}")
print(f"Model's predicted class: {class_names_adult[predicted_class_idx_lime]}")

# %%
# Generate explanation
explanation_lime = explainer_lime.explain_instance(
    data_row=instance_to_explain_lime_np,
    predict_fn=pytorch_predict_proba_for_lime,
    num_features=10, 
    top_labels=1
)

# %%
# explanation.show_in_notebook() relies on an import removed in IPython 9, so render its HTML directly
display(HTML(explanation_lime.as_html(show_table=True, show_all=False)))

print("LIME explanation (feature: weight contributing to predicted class):")
for feat_idx, weight in explanation_lime.as_list(label=predicted_class_idx_lime):
    print(f"  {feat_idx}: {weight:.3f}")


# %% [markdown]
# ## SHAP (SHapley Additive exPlanations)
#
# <p>SHAP assigns each feature an importance value for a particular prediction. It's based on game theory and provides a unified measure of feature importance.</p>

# %%
def pytorch_predict_for_shap(tensor_data_shap):
    adult_model.eval()
    if isinstance(tensor_data_shap, np.ndarray):
        tensor_data_shap = torch.from_numpy(tensor_data_shap).float().to(device)
    elif tensor_data_shap.device != device:
        tensor_data_shap = tensor_data_shap.to(device)

    with torch.no_grad():
        logits = adult_model(tensor_data_shap)
    return logits


# %%
background_data_shap = X_train_adult[torch.randperm(X_train_adult.size(0))[:100]]
background_data_shap

# %%
explainer_shap = shap.GradientExplainer(adult_model, background_data_shap)

# %%
# Explain predictions on a few test instances
num_shap_instances = 5
instances_to_explain_shap_tensor = X_test_adult[:num_shap_instances].to(device)

# %%
shap_values_output = explainer_shap.shap_values(instances_to_explain_shap_tensor)

# %% [markdown]
# For binary classification with GradientExplainer on logits, shap_values might be a list of [shap_values_for_class_0, shap_values_for_class_1]
# or just shap_values for the positive class if the model output is single logit.

# %%
type(shap_values_output)

# %%
if isinstance(shap_values_output, list): 
    if len(shap_values_output) == 1:
        shap_values_to_plot_data = shap_values_output[0]
    else:
        print(f"SHAP returned a list of {len(shap_values_output)} items. Assuming explanation for class 1 (index 1 if multi-logit).")
        shap_values_to_plot_data = shap_values_output[1]
elif isinstance(shap_values_output, np.ndarray) or isinstance(shap_values_output, torch.Tensor):
    shap_values_to_plot_data = shap_values_output
elif hasattr(shap_values_output, 'values') and hasattr(shap_values_output, 'base_values'):
    shap_values_to_plot_data = shap_values_output.values
else:
    raise TypeError(f"Unexpected type for shap_values_output: {type(shap_values_output)}")

# %%
# Convert to NumPy if they are tensors
if isinstance(shap_values_to_plot_data, torch.Tensor):
    shap_values_to_plot_np = shap_values_to_plot_data.cpu().numpy()
else:
    shap_values_to_plot_np = np.array(shap_values_to_plot_data) # Ensure it's a NumPy array

instances_to_explain_shap_np = instances_to_explain_shap_tensor.cpu().numpy()

# %%
base_value_for_plots = None
if hasattr(explainer_shap, 'expected_value'):
    base_value_for_plots = explainer_shap.expected_value
    if isinstance(base_value_for_plots, torch.Tensor):
        base_value_for_plots = base_value_for_plots.cpu().item()
    elif isinstance(base_value_for_plots, np.ndarray):
        base_value_for_plots = base_value_for_plots.item() 
    print(f"Using explainer_shap.expected_value: {base_value_for_plots}")
elif isinstance(shap_values_output, shap.Explanation) and hasattr(shap_values_output, 'base_values'):
    if shap_values_output.base_values.ndim > 0 and shap_values_output.base_values.shape[0] > 0:
        base_value_for_plots = shap_values_output.base_values[0]
    else:
        base_value_for_plots = shap_values_output.base_values # If already scalar
    if isinstance(base_value_for_plots, np.ndarray) and base_value_for_plots.size == 1:
        base_value_for_plots = base_value_for_plots.item()
    print(f"Using base_values from SHAP Explanation object: {base_value_for_plots}")
else:
    print("Warning: Could not automatically determine SHAP base value. Force/Waterfall plots might be affected or use a default.")

# %%
print("\nSHAP Summary Plot:")
shap.summary_plot(shap_values_to_plot_np, instances_to_explain_shap_np,
                  feature_names=feature_names_processed, show=False)
plt.title("SHAP Summary Plot")
plt.show()

# %%
