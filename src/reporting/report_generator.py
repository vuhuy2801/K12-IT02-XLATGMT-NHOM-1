import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.reporting.visualizer import ModelPerformanceVisualizer

class PerformanceReport:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = ModelPerformanceVisualizer(output_dir)
        
    def generate_training_report(self, 
                               history: Dict[str, Dict[str, List[float]]], 
                               y_true: list,
                               y_pred: list,
                               classes: list,
                               experiment_name: str = None):
        """Tạo báo cáo training đầy đủ với visualizations
        
        Args:
            history: Dictionary chứa metrics training
            y_true: Ground truth labels từ validation set
            y_pred: Predicted labels từ validation set
            classes: Tên các classes
            experiment_name: Tên experiment
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Tạo thư mục cho experiment
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)
        
        try:
            # Generate và save các plots
            metrics_plot = self.visualizer.plot_metrics_over_time(history)
            cm_plot_counts, cm_plot_percent = self.visualizer.plot_confusion_matrix(
                y_true, y_pred, classes
            )
            report_dict = self.visualizer.generate_classification_report(
                y_true, y_pred, classes
            )
            
            # Save metrics history
            metrics_path = experiment_dir / "training_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'history': history,
                    'final_metrics': report_dict,
                    'plots': {
                        'metrics_over_time': metrics_plot,
                        'confusion_matrix_counts': cm_plot_counts,
                        'confusion_matrix_percent': cm_plot_percent
                    }
                }, f, indent=4)
            
            # Generate summary report
            summary_path = experiment_dir / "training_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Training Summary - {experiment_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Final Metrics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Accuracy: {report_dict['accuracy']:.4f}\n")
                f.write(f"Macro Avg F1: {report_dict['macro avg']['f1-score']:.4f}\n")
                f.write(f"Weighted Avg F1: {report_dict['weighted avg']['f1-score']:.4f}\n\n")
                
                f.write("Per-Class Metrics:\n")
                f.write("-" * 20 + "\n")
                for class_name in classes:
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {report_dict[class_name]['precision']:.4f}\n")
                    f.write(f"  Recall: {report_dict[class_name]['recall']:.4f}\n")
                    f.write(f"  F1-score: {report_dict[class_name]['f1-score']:.4f}\n")
            
            logger.info(f"Training report generated at: {experiment_dir}")
            return experiment_dir
            
        except Exception as e:
            logger.error(f"Error generating training report: {str(e)}")
            raise