"""

analysis_and_evaluation.py

Usage: run from project root (where `MP_Data` and `model.h5` live).
This script computes:
- Original class distribution (counts of sequences per action)
- Balanced class distribution (after simple undersampling to match min class)
- Feature correlation heatmap (per-feature correlation using sequence-level mean features)
- Loads `model.h5` (Keras) and evaluates on a held-out test split: prints loss and accuracy
- Confusion matrix and classification report

Outputs are saved to `analysis/` as PNGs and CSVs.

Examples:
    python analysis_and_evaluation.py --mode all
    python analysis_and_evaluation.py --mode distributions
    python analysis_and_evaluation.py --mode evaluate --model model.h5

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report as sk_classification_report

try:
    from function import DATA_PATH, actions, sequence_length
except Exception:
    # fallback defaults
    DATA_PATH = 'MP_Data'
    actions = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
    sequence_length = 30

ANALYSIS_DIR = 'analysis'
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_sequences(return_label_map=True):
    """Load sequences as arrays of shape (sequence_length, feature_dim).
    For each sequence, compute the mean across frames to get a single feature vector.
    Returns X (n_sequences, feature_dim), y (n_sequences,), label_map dict.
    """
    sequences = []
    labels = []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        for sequence in os.listdir(os.path.join(DATA_PATH, action)) if os.path.isdir(os.path.join(DATA_PATH, action)) else []:
            seq_path = os.path.join(DATA_PATH, action, sequence)
            # collect frame files sorted
            frame_files = [os.path.join(seq_path, f) for f in sorted(os.listdir(seq_path)) if f.endswith('.npy')]
            if len(frame_files) < sequence_length:
                # incomplete sequence, skip
                continue
            try:
                frames = [np.load(f) for f in frame_files[:sequence_length]]
            except Exception as e:
                print(f"Warning: failed to load sequence {action}/{sequence}: {e}")
                continue
            # stack and compute mean feature vector for the sequence
            frames = np.stack(frames)
            mean_feat = frames.mean(axis=0)
            sequences.append(mean_feat)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = np.array(labels)
    if return_label_map:
        return X, y, label_map
    return X, y


def compute_class_distributions():
    # Count original sequences per action (folders with enough frames)
    counts = {}
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        count = 0
        if os.path.isdir(action_dir):
            for seq in os.listdir(action_dir):
                seq_path = os.path.join(action_dir, seq)
                if not os.path.isdir(seq_path):
                    continue
                frame_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
                if len(frame_files) >= sequence_length:
                    count += 1
        counts[action] = count

    df = pd.DataFrame.from_dict(counts, orient='index', columns=['count'])
    df.index.name = 'action'
    df.to_csv(os.path.join(ANALYSIS_DIR, 'original_class_distribution.csv'))

    # Plot
    plt.figure(figsize=(12,6))
    sns.barplot(x=df.index, y='count', data=df.reset_index())
    plt.title('Original class distribution (sequences per action)')
    plt.xlabel('Action')
    plt.ylabel('Sequence count')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'original_class_distribution.png'))
    plt.close()

    # Balanced distribution (undersample to min count)
    min_count = int(df['count'].min()) if len(df)>0 else 0
    balanced = df.copy()
    balanced['balanced_count'] = min_count
    balanced.to_csv(os.path.join(ANALYSIS_DIR, 'balanced_class_distribution.csv'))

    plt.figure(figsize=(12,6))
    sns.barplot(x=balanced.index, y='balanced_count', data=balanced.reset_index())
    plt.title('Balanced class distribution (undersampled)')
    plt.xlabel('Action')
    plt.ylabel('Sequence count (balanced)')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'balanced_class_distribution.png'))
    plt.close()

    print('Saved original and balanced class distributions to', ANALYSIS_DIR)


def compute_feature_correlation(X):
    # X is (n_samples, n_features)
    df = pd.DataFrame(X)
    corr = df.corr()
    corr.to_csv(os.path.join(ANALYSIS_DIR, 'feature_correlation.csv'))

    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='vlag', center=0)
    plt.title('Feature correlation (mean sequence-level features)')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'feature_correlation.png'))
    plt.close()
    print('Saved feature correlation heatmap')


def evaluate_model(model_path, test_size=0.05):
    from tensorflow.keras.models import load_model

    X, y, label_map = load_sequences()
    if X.size == 0:
        print('No sequences found. Ensure MP_Data is populated.')
        return

    # One-hot expected by model, but model.evaluate accepts categorical or not depending on compile
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # load model
    if not os.path.isfile(model_path):
        print('Model file not found:', model_path)
        return
    model = load_model(model_path)

    # If model expects sequences (30,63), but we pass mean vector (63,), reshape accordingly.
    # Try to infer model input shape
    input_shape = model.input_shape
    print('Model input shape:', input_shape)

    # If model expects time dimension, expand X_test accordingly by repeating mean vector
    if len(input_shape) == 3:
        time_steps = input_shape[1]
        # expand each mean feature to shape (time_steps, n_features) by repeating
        X_test_seq = np.repeat(X_test[:, np.newaxis, :], time_steps, axis=1)
    else:
        X_test_seq = X_test

    # Convert labels to categorical
    from tensorflow.keras.utils import to_categorical
    y_test_cat = to_categorical(y_test, num_classes=actions.shape[0])

    # Evaluate
    loss, acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    print(f'Evaluation results -- Loss: {loss:.4f}, Accuracy: {acc:.4f}')

    # Predictions
    y_pred_probs = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=actions, columns=actions)
    cm_df.to_csv(os.path.join(ANALYSIS_DIR, 'confusion_matrix.csv'))

    plt.figure(figsize=(12,10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'confusion_matrix.png'))
    plt.close()

    # Classification report (as CSV and bar plot)
    report_dict = sk_classification_report(y_test, y_pred, target_names=list(actions), output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(ANALYSIS_DIR, 'classification_report.csv'))

    # Plot precision/recall/f1 for each class (only the actions rows)
    try:
        class_rows = [a for a in list(actions) if a in report_df.index]
        metrics_plot_df = report_df.loc[class_rows, ['precision', 'recall', 'f1-score']]
        plt.figure(figsize=(14,7))
        metrics_plot_df.plot(kind='bar')
        plt.title('Per-class Precision / Recall / F1-score')
        plt.ylabel('Score')
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'classification_report_metrics.png'))
        plt.close()
    except Exception as e:
        print('Could not plot per-class metrics:', e)

    # Save accuracy and loss as a small image for quick viewing
    # Plot training & validation accuracy and loss if training history exists
    try:
        import json
        hist_path = os.path.join(ANALYSIS_DIR, 'training_history.json')
        if os.path.isfile(hist_path):
            with open(hist_path, 'r') as f:
                history = json.load(f)

            # history expected to contain keys like 'loss','val_loss','categorical_accuracy','val_categorical_accuracy'
            loss_hist = history.get('loss') or history.get('training_loss')
            val_loss_hist = history.get('val_loss')
            acc_hist = history.get('categorical_accuracy') or history.get('accuracy')
            val_acc_hist = history.get('val_categorical_accuracy') or history.get('val_accuracy')

            epochs = range(len(loss_hist)) if loss_hist is not None else range(len(acc_hist))

            fig, axes = plt.subplots(1, 2, figsize=(16,4))
            # Accuracy subplot
            axes[0].plot(epochs, acc_hist, label='Training Accuracy')
            if val_acc_hist is not None:
                axes[0].plot(epochs, val_acc_hist, label='Validation Accuracy')
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()

            # Loss subplot
            axes[1].plot(epochs, loss_hist, label='Training Loss')
            if val_loss_hist is not None:
                axes[1].plot(epochs, val_loss_hist, label='Validation Loss')
            axes[1].set_title('Model Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(ANALYSIS_DIR, 'accuracy_loss_full.png'))
            plt.close(fig)
            print('Saved training/validation accuracy and loss to analysis/accuracy_loss_full.png')
                # Also save separate learning curves for accuracy and loss
                try:
                    # Accuracy curve
                    plt.figure(figsize=(8,4))
                    plt.plot(epochs, acc_hist, label='Training Accuracy')
                    if val_acc_hist is not None:
                        plt.plot(epochs, val_acc_hist, label='Validation Accuracy')
                    plt.title('Learning Curve - Accuracy')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.ylim(0, 1.02)
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(ANALYSIS_DIR, 'learning_curve_accuracy.png'))
                    plt.close()

                    # Loss curve
                    plt.figure(figsize=(8,4))
                    plt.plot(epochs, loss_hist, label='Training Loss')
                    if val_loss_hist is not None:
                        plt.plot(epochs, val_loss_hist, label='Validation Loss')
                    plt.title('Learning Curve - Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(ANALYSIS_DIR, 'learning_curve_loss.png'))
                    plt.close()
                    print('Saved separate learning curves to analysis/learning_curve_accuracy.png and analysis/learning_curve_loss.png')
                except Exception as e:
                    print('Could not save separate learning curves:', e)
        else:
            # no training history available
            print('No training history file found at', hist_path)
    except Exception as e:
        print('Could not generate full accuracy/loss plot:', e)

    print('\nClassification Report saved to CSV and per-class plot')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['all', 'distributions', 'correlation', 'evaluate'], default='all')
    parser.add_argument('--model', default='model.h5', help='Path to Keras model file for evaluation')
    args = parser.parse_args()

    if args.mode in ('all', 'distributions'):
        compute_class_distributions()

    if args.mode in ('all', 'correlation'):
        X, y, _ = load_sequences()
        if X.size == 0:
            print('No sequence-level features found to compute correlations. Run data generation first.')
        else:
            compute_feature_correlation(X)

    if args.mode in ('all', 'evaluate'):
        evaluate_model(args.model)


if __name__ == '__main__':
    main()
