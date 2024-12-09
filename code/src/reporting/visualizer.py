import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ModelPerformanceVisualizer:
    def __init__(self, checkpoint_dir: str):
        self.figure_dir = Path(checkpoint_dir) / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics_over_time(self, history: dict) -> str:
        """Vẽ biểu đồ các metrics theo thời gian
        
        Args:
            history: Dictionary chứa metrics theo epoch
            
        Returns:
            Path to saved figure
        """
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot accuracy
            plt.subplot(2, 2, 1)
            plt.plot(history['train_accuracy'], 'b-', label='Train')
            plt.plot(history['val_accuracy'], 'r--', label='Validation')
            plt.title('Accuracy over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            # Plot precision
            plt.subplot(2, 2, 2)
            for class_name, values in history['precision'].items():
                plt.plot(values, label=f'{class_name}')
            plt.title('Precision over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True)
            
            # Plot recall
            plt.subplot(2, 2, 3)
            for class_name, values in history['recall'].items():
                plt.plot(values, label=f'{class_name}')
            plt.title('Recall over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True)
            
            # Plot F1-score
            plt.subplot(2, 2, 4)
            for class_name, values in history['f1'].items():
                plt.plot(values, label=f'{class_name}')
            plt.title('F1 Score over Time')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save figure
            fig_path = self.figure_dir / "metrics_over_time.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(fig_path)
            
        except Exception as e:
            logger.error(f"Error plotting metrics over time: {str(e)}")
            raise

    def plot_confusion_matrix(self, y_true: list, y_pred: list, classes: list) -> str:
        """Vẽ confusion matrix với annotations
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            classes: Tên các classes
            
        Returns:
            Path to saved figure
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            # Tính percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=classes, yticklabels=classes)
            
            # Thêm title và labels
            plt.title('Confusion Matrix\n(values in counts)', pad=20)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save first figure
            fig_path_counts = self.figure_dir / "confusion_matrix_counts.png"
            plt.savefig(fig_path_counts, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot percentage version
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                       xticklabels=classes, yticklabels=classes)
            
            plt.title('Confusion Matrix\n(values in percentages)', pad=20)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save second figure
            fig_path_percent = self.figure_dir / "confusion_matrix_percent.png"
            plt.savefig(fig_path_percent, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(fig_path_counts), str(fig_path_percent)
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise

    def generate_classification_report(self, y_true: list, y_pred: list, classes: list) -> dict:
        """Tạo báo cáo classification metrics chi tiết
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            classes: Tên các classes
            
        Returns:
            Dictionary chứa các metrics
        """
        try:
            # Tính toán metrics
            report_dict = classification_report(y_true, y_pred, 
                                             target_names=classes,
                                             output_dict=True)
            
            # Save detailed report as text
            report_path = self.figure_dir / "classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(classification_report(y_true, y_pred, target_names=classes))
                f.write("\n\nDetailed Metrics:\n")
                f.write("-" * 50 + "\n")
                
                # Add more detailed analysis
                for class_name in classes:
                    f.write(f"\nClass: {class_name}\n")
                    f.write(f"Precision: {report_dict[class_name]['precision']:.4f}\n")
                    f.write(f"Recall: {report_dict[class_name]['recall']:.4f}\n")
                    f.write(f"F1-score: {report_dict[class_name]['f1-score']:.4f}\n")
                    f.write(f"Support: {report_dict[class_name]['support']}\n")
                    
                # Add macro and weighted averages
                f.write("\nOverall Metrics:\n")
                f.write(f"Macro Avg F1: {report_dict['macro avg']['f1-score']:.4f}\n")
                f.write(f"Weighted Avg F1: {report_dict['weighted avg']['f1-score']:.4f}\n")
            
            return report_dict
            
        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
            raise