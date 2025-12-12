"""
Evaluate the trained inverter model and generate attack analysis report.
"""

import torch
import pickle
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from attack.inverter_model import EmbeddingInverter, LSTMInverter, AttentionInverter
from evaluation.metrics import InversionMetrics


class AttackEvaluator:
    """Evaluate embedding inversion attack."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        tokenizer_name: str = "neuralmind/bert-base-portuguese-cased",
        device: str = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            model_type: Type of model ('mlp', 'lstm', 'attention')
            tokenizer_name: Name of tokenizer
            device: Device to use
        """
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model architecture
        vocab_size = self.tokenizer.vocab_size
        embedding_dim = 768  # BERT-base dimension
        
        if model_type == 'mlp':
            self.model = EmbeddingInverter(
                embedding_dim=embedding_dim,
                vocab_size=vocab_size,
                max_seq_length=128
            )
        elif model_type == 'lstm':
            self.model = LSTMInverter(
                embedding_dim=embedding_dim,
                vocab_size=vocab_size,
                max_seq_length=128
            )
        elif model_type == 'attention':
            self.model = AttentionInverter(
                embedding_dim=embedding_dim,
                vocab_size=vocab_size,
                max_seq_length=128
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.metrics_calculator = InversionMetrics()
        
        print(f"Model loaded successfully on {self.device}")
    
    def reconstruct_texts(
        self,
        embeddings: np.ndarray,
        batch_size: int = 32
    ) -> List[str]:
        """
        Reconstruct texts from embeddings.
        
        Args:
            embeddings: Input embeddings
            batch_size: Batch size for processing
            
        Returns:
            List of reconstructed texts
        """
        reconstructed_texts = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(embeddings), batch_size), desc="Reconstructing"):
                batch_embeddings = torch.FloatTensor(
                    embeddings[i:i+batch_size]
                ).to(self.device)
                
                # Get predictions
                logits = self.model(batch_embeddings)
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode to text
                for ids in predicted_ids:
                    text = self.tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    reconstructed_texts.append(text)
        
        return reconstructed_texts
    
    def evaluate_on_dataset(
        self,
        embeddings_path: str,
        num_samples: int = None
    ) -> Dict:
        """
        Evaluate attack on a dataset.
        
        Args:
            embeddings_path: Path to embeddings file
            num_samples: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\nLoading embeddings from {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = data['embeddings']
        original_texts = data['texts']
        
        if num_samples:
            embeddings = embeddings[:num_samples]
            original_texts = original_texts[:num_samples]
        
        print(f"Evaluating on {len(embeddings)} samples...")
        
        # Reconstruct texts
        reconstructed_texts = self.reconstruct_texts(embeddings)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = self.metrics_calculator.evaluate_batch(
            reconstructed_texts,
            original_texts
        )
        
        # Store examples
        examples = []
        for i in range(min(10, len(original_texts))):
            examples.append({
                'original': original_texts[i],
                'reconstructed': reconstructed_texts[i]
            })
        
        results = {
            'metrics': metrics,
            'examples': examples,
            'num_samples': len(embeddings),
            'model_type': self.model_type
        }
        
        return results
    
    def generate_report(
        self,
        results: Dict,
        output_dir: str = 'results'
    ):
        """
        Generate comprehensive attack analysis report.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = results['metrics']
        
        # Print metrics
        self.metrics_calculator.print_metrics(metrics)
        
        # Save metrics to JSON
        metrics_path = os.path.join(output_dir, 'attack_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved metrics to {metrics_path}")
        
        # Save examples
        examples_path = os.path.join(output_dir, 'reconstruction_examples.txt')
        with open(examples_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EMBEDDING INVERSION ATTACK - RECONSTRUCTION EXAMPLES\n")
            f.write("="*80 + "\n\n")
            
            for i, example in enumerate(results['examples'], 1):
                f.write(f"Example {i}:\n")
                f.write("-" * 80 + "\n")
                f.write(f"ORIGINAL:\n{example['original']}\n\n")
                f.write(f"RECONSTRUCTED:\n{example['reconstructed']}\n")
                f.write("="*80 + "\n\n")
        
        print(f"✓ Saved examples to {examples_path}")
        
        # Create visualizations
        self._create_visualizations(metrics, output_dir)
        
        # Generate markdown report
        self._generate_markdown_report(results, output_dir)
    
    def _create_visualizations(self, metrics: Dict, output_dir: str):
        """Create visualization plots."""
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Text similarity metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        similarity_metrics = {
            'BLEU-1': metrics.get('bleu1', 0),
            'BLEU-2': metrics.get('bleu2', 0),
            'BLEU-4': metrics.get('bleu4', 0),
            'ROUGE-1': metrics.get('rouge1_f', 0),
            'ROUGE-2': metrics.get('rouge2_f', 0),
            'ROUGE-L': metrics.get('rougeL_f', 0),
            'Similarity': metrics.get('similarity', 0)
        }
        
        bars = ax.bar(similarity_metrics.keys(), similarity_metrics.values(), 
                     color='steelblue', alpha=0.8)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Text Similarity Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'similarity_metrics.png'), dpi=300)
        plt.close()
        
        # 2. Keyword recovery metrics
        fig, ax = plt.subplots(figsize=(8, 6))
        keyword_metrics = {
            'Precision': metrics.get('keyword_precision', 0),
            'Recall': metrics.get('keyword_recall', 0),
            'F1-Score': metrics.get('keyword_f1', 0)
        }
        
        bars = ax.bar(keyword_metrics.keys(), keyword_metrics.values(),
                     color=['#2ecc71', '#e74c3c', '#3498db'], alpha=0.8)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Keyword Recovery Performance', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'keyword_recovery.png'), dpi=300)
        plt.close()
        
        # 3. Risk assessment gauge
        fig, ax = plt.subplots(figsize=(10, 6))
        
        risk_score = (
            metrics.get('similarity', 0) * 0.3 +
            metrics.get('keyword_f1', 0) * 0.4 +
            metrics.get('overall_leakage_rate', 0) * 0.3
        )
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        risk_levels = ['Low Risk\n(0.0-0.3)', 'Medium Risk\n(0.3-0.6)', 'High Risk\n(0.6-1.0)']
        
        ax.barh(risk_levels, [0.3, 0.3, 0.4], color=colors, alpha=0.3)
        ax.barh(['Current Risk'], [risk_score], color='darkred', alpha=0.8, height=0.3)
        
        ax.set_xlabel('Risk Score', fontsize=12)
        ax.set_title('Privacy Risk Assessment', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.text(risk_score + 0.02, 0, f'{risk_score:.3f}', 
               va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'risk_assessment.png'), dpi=300)
        plt.close()
        
        print(f"✓ Saved visualizations to {plots_dir}")
    
    def _generate_markdown_report(self, results: Dict, output_dir: str):
        """Generate markdown report."""
        report_path = os.path.join(output_dir, 'ATTACK_REPORT.md')
        metrics = results['metrics']
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Análise de Ataque de Inversão de Embeddings\n\n")
            f.write("## 📊 Resumo Executivo\n\n")
            f.write(f"- **Modelo Atacante**: {results['model_type'].upper()}\n")
            f.write(f"- **Amostras Avaliadas**: {results['num_samples']}\n")
            f.write(f"- **Modelo Alvo**: BERTimbau (BERT-base Portuguese)\n\n")
            
            f.write("## 🎯 Métricas de Similaridade Textual\n\n")
            f.write("| Métrica | Score |\n")
            f.write("|---------|-------|\n")
            f.write(f"| BLEU-1 | {metrics.get('bleu1', 0):.4f} |\n")
            f.write(f"| BLEU-2 | {metrics.get('bleu2', 0):.4f} |\n")
            f.write(f"| BLEU-4 | {metrics.get('bleu4', 0):.4f} |\n")
            f.write(f"| ROUGE-1 | {metrics.get('rouge1_f', 0):.4f} |\n")
            f.write(f"| ROUGE-2 | {metrics.get('rouge2_f', 0):.4f} |\n")
            f.write(f"| ROUGE-L | {metrics.get('rougeL_f', 0):.4f} |\n")
            f.write(f"| Similaridade | {metrics.get('similarity', 0):.4f} |\n\n")
            
            f.write("## 🔑 Recuperação de Palavras-Chave\n\n")
            f.write("| Métrica | Score |\n")
            f.write("|---------|-------|\n")
            f.write(f"| Precisão | {metrics.get('keyword_precision', 0):.4f} |\n")
            f.write(f"| Recall | {metrics.get('keyword_recall', 0):.4f} |\n")
            f.write(f"| F1-Score | {metrics.get('keyword_f1', 0):.4f} |\n\n")
            
            f.write("## ⚠️ Vazamento de Informações Sensíveis\n\n")
            f.write(f"**Taxa de Vazamento Geral**: {metrics.get('overall_leakage_rate', 0):.2%}\n\n")
            
            f.write("## 🔒 Avaliação de Risco\n\n")
            risk_score = (
                metrics.get('similarity', 0) * 0.3 +
                metrics.get('keyword_f1', 0) * 0.4 +
                metrics.get('overall_leakage_rate', 0) * 0.3
            )
            
            if risk_score < 0.3:
                risk_level = "🟢 BAIXO"
                recommendation = "Os embeddings apresentam baixo risco de reconstrução."
            elif risk_score < 0.6:
                risk_level = "🟡 MÉDIO"
                recommendation = "Recomenda-se implementar medidas adicionais de proteção."
            else:
                risk_level = "🔴 ALTO"
                recommendation = "CRÍTICO: Os embeddings são vulneráveis a ataques de inversão."
            
            f.write(f"**Nível de Risco**: {risk_level}\n\n")
            f.write(f"**Score de Risco**: {risk_score:.3f}\n\n")
            f.write(f"**Recomendação**: {recommendation}\n\n")
            
            f.write("## 📈 Visualizações\n\n")
            f.write("![Similarity Metrics](plots/similarity_metrics.png)\n\n")
            f.write("![Keyword Recovery](plots/keyword_recovery.png)\n\n")
            f.write("![Risk Assessment](plots/risk_assessment.png)\n\n")
            
            f.write("## 💡 Conclusões\n\n")
            f.write("1. **Viabilidade do Ataque**: ")
            if metrics.get('similarity', 0) > 0.5:
                f.write("O ataque demonstrou ser viável, com reconstrução parcial do texto original.\n")
            else:
                f.write("O ataque teve sucesso limitado na reconstrução do texto original.\n")
            
            f.write("2. **Privacidade dos Dados**: ")
            if metrics.get('overall_leakage_rate', 0) > 0.3:
                f.write("Há risco significativo de vazamento de informações sensíveis.\n")
            else:
                f.write("O risco de vazamento de informações sensíveis é controlado.\n")
            
            f.write("3. **Recomendações de Segurança**:\n")
            f.write("   - Implementar técnicas de differential privacy\n")
            f.write("   - Adicionar ruído aos embeddings antes do armazenamento\n")
            f.write("   - Utilizar criptografia homomórfica para processamento seguro\n")
            f.write("   - Realizar auditorias regulares de segurança\n\n")
        
        print(f"✓ Saved report to {report_path}")


def main():
    """Main evaluation function."""
    
    # Configuration
    MODEL_PATH = "models/attacker/mlp/best_inverter.pt"
    MODEL_TYPE = "mlp"
    EMBEDDINGS_PATH = "data/embeddings/test_embeddings.pkl"
    OUTPUT_DIR = "results"
    NUM_SAMPLES = None  # Evaluate all samples
    
    print("="*60)
    print("EMBEDDING INVERSION ATTACK EVALUATION")
    print("="*60)
    
    # Create evaluator
    evaluator = AttackEvaluator(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE
    )
    
    # Evaluate
    results = evaluator.evaluate_on_dataset(
        embeddings_path=EMBEDDINGS_PATH,
        num_samples=NUM_SAMPLES
    )
    
    # Generate report
    evaluator.generate_report(results, output_dir=OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()