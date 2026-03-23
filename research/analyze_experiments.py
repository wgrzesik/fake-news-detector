import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ExperimentAnalyzer:
    """Analyze experiment results"""

    def __init__(self, summary_csv: str):
        self.df = pd.read_csv(summary_csv)

    def get_best_experiments(self, top_n: int = 10) -> pd.DataFrame:
        """Get top experiments by F1 score"""
        return self.df.nlargest(top_n, 'f1_score')

    def get_worst_experiments(self, bottom_n: int = 5) -> pd.DataFrame:
        """Get worst experiments by F1 score"""
        return self.df.nsmallest(bottom_n, 'f1_score')

    def get_by_dataset(self, dataset: str) -> pd.DataFrame:
        """Filter by dataset"""
        return self.df[self.df['dataset'] == dataset].sort_values('f1_score', ascending=False)

    def get_by_model(self, model: str) -> pd.DataFrame:
        """Filter by model"""
        return self.df[self.df['model'] == model].sort_values('f1_score', ascending=False)

    def get_by_embedding(self, embedding: str) -> pd.DataFrame:
        """Filter by embedding"""
        return self.df[self.df['embedding'] == embedding].sort_values('f1_score', ascending=False)

    def dataset_summary(self) -> pd.DataFrame:
        """Summary statistics by dataset"""
        return self.df.groupby('dataset').agg({
            'f1_score': ['count', 'mean', 'max', 'min', 'std'],
            'accuracy': ['mean', 'max'],
            'train_time_sec': ['sum', 'mean']
        }).round(4)

    def model_summary(self) -> pd.DataFrame:
        """Summary statistics by model"""
        return self.df.groupby('model').agg({
            'f1_score': ['count', 'mean', 'max', 'min', 'std'],
            'accuracy': ['mean', 'max']
        }).round(4).sort_values(('f1_score', 'mean'), ascending=False)

    def embedding_summary(self) -> pd.DataFrame:
        """Summary statistics by embedding"""
        return self.df.groupby('embedding').agg({
            'f1_score': ['count', 'mean', 'max', 'min', 'std'],
            'accuracy': ['mean', 'max']
        }).round(4).sort_values(('f1_score', 'mean'), ascending=False)

    def combo_summary(self) -> pd.DataFrame:
        """Summary by model-embedding combination"""
        return self.df.groupby(['model', 'embedding']).agg({
            'f1_score': ['count', 'mean', 'max'],
            'accuracy': 'mean'
        }).round(4).sort_values(('f1_score', 'mean'), ascending=False)

    def plot_f1_by_dataset(self, output: str = "f1_by_dataset.png"):
        """Plot F1 scores by dataset"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df.boxplot(column='f1_score', by='dataset', ax=ax)
        plt.title('F1 Score Distribution by Dataset')
        plt.suptitle('')
        plt.ylabel('F1 Score')
        plt.xlabel('Dataset')
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")

    def plot_f1_by_model(self, output: str = "f1_by_model.png"):
        """Plot F1 scores by model"""
        fig, ax = plt.subplots(figsize=(12, 6))
        self.df.boxplot(column='f1_score', by='model', ax=ax)
        plt.title('F1 Score Distribution by Model')
        plt.suptitle('')
        plt.ylabel('F1 Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")

    def plot_heatmap(self, output: str = "model_embedding_heatmap.png"):
        """Plot heatmap of F1 scores for model-embedding combinations"""
        pivot = self.df.pivot_table(
            values='f1_score',
            index='model',
            columns='embedding',
            aggfunc='max'
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGn', ax=ax, cbar_kws={'label': 'F1 Score'})
        plt.title('Max F1 Score: Model vs Embedding')
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")

    def plot_training_time(self, output: str = "training_time.png"):
        """Plot training time by model"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df.boxplot(column='train_time_sec', by='model', ax=ax)
        plt.title('Training Time by Model')
        plt.suptitle('')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")

    def generate_report(self, output_path: str = "report.txt"):
        """Generate comprehensive text report"""
        with open(output_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("EXPERIMENT ANALYSIS REPORT\n")
            f.write("="*100 + "\n\n")

            f.write("OVERALL STATISTICS\n")
            f.write("-"*100 + "\n")
            f.write(f"Total experiments: {len(self.df)}\n")
            f.write(f"Average F1 Score: {self.df['f1_score'].mean():.4f}\n")
            f.write(f"Std F1 Score: {self.df['f1_score'].std():.4f}\n")
            f.write(f"Best F1 Score: {self.df['f1_score'].max():.4f}\n")
            f.write(f"Worst F1 Score: {self.df['f1_score'].min():.4f}\n\n")

            f.write("TOP 10 EXPERIMENTS\n")
            f.write("-"*100 + "\n")
            f.write(self.get_best_experiments(10).to_string(index=False))
            f.write("\n\n")

            f.write("BY DATASET\n")
            f.write("-"*100 + "\n")
            f.write(str(self.dataset_summary()))
            f.write("\n\n")

            f.write("BY MODEL\n")
            f.write("-"*100 + "\n")
            f.write(str(self.model_summary()))
            f.write("\n\n")

            f.write("BY EMBEDDING\n")
            f.write("-"*100 + "\n")
            f.write(str(self.embedding_summary()))
            f.write("\n\n")

            f.write("MODEL-EMBEDDING COMBINATIONS\n")
            f.write("-"*100 + "\n")
            f.write(str(self.combo_summary()))
            f.write("\n")

        print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--summary', required=True, help='Path to SUMMARY.csv')
    parser.add_argument('--dataset', help='Filter by dataset')
    parser.add_argument('--model', help='Filter by model')
    parser.add_argument('--embedding', help='Filter by embedding')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    parser.add_argument('--report', help='Output report path')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top experiments to show')
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(args.summary)

    print("\n" + "="*100)
    print("EXPERIMENT ANALYSIS")
    print("="*100 + "\n")

    # Apply filters
    df = analyzer.df
    if args.dataset:
        df = analyzer.get_by_dataset(args.dataset)
        print(f"[Filtered by dataset: {args.dataset}] ({len(df)} experiments)\n")
    if args.model:
        df = analyzer.get_by_model(args.model)
        print(f"[Filtered by model: {args.model}] ({len(df)} experiments)\n")
    if args.embedding:
        df = analyzer.get_by_embedding(args.embedding)
        print(f"[Filtered by embedding: {args.embedding}] ({len(df)} experiments)\n")

    # Print top experiments
    print(f"TOP {args.top_n} EXPERIMENTS\n")
    print(df.nlargest(args.top_n, 'f1_score')[[
        'dataset', 'model', 'embedding', 'f1_score', 'accuracy'
    ]].to_string(index=False))

    print("\n" + "-"*100 + "\n")
    print("DATASET SUMMARY\n")
    print(analyzer.dataset_summary())

    print("\n" + "-"*100 + "\n")
    print("MODEL SUMMARY\n")
    print(analyzer.model_summary())

    print("\n" + "-"*100 + "\n")
    print("EMBEDDING SUMMARY\n")
    print(analyzer.embedding_summary())

    # Generate plots
    if args.plots:
        print("\n[Generating plots...]")
        analyzer.plot_f1_by_dataset()
        analyzer.plot_f1_by_model()
        analyzer.plot_heatmap()
        analyzer.plot_training_time()

    # Generate report
    if args.report:
        analyzer.generate_report(args.report)

    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()