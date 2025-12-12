"""
Evaluation metrics for embedding inversion attack.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from difflib import SequenceMatcher


class InversionMetrics:
    """Calculate various metrics for evaluating inversion attack success."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction()
    
    def calculate_token_accuracy(
        self,
        predicted_ids: torch.Tensor,
        target_ids: torch.Tensor,
        pad_token_id: int
    ) -> float:
        """
        Calculate token-level accuracy.
        
        Args:
            predicted_ids: Predicted token IDs
            target_ids: Ground truth token IDs
            pad_token_id: ID of padding token to ignore
            
        Returns:
            Token accuracy
        """
        mask = target_ids != pad_token_id
        correct = (predicted_ids == target_ids) & mask
        accuracy = correct.sum().item() / mask.sum().item()
        return accuracy
    
    def calculate_bleu(
        self,
        predicted_text: str,
        reference_text: str
    ) -> Dict[str, float]:
        """
        Calculate BLEU scores.
        
        Args:
            predicted_text: Predicted text
            reference_text: Reference text
            
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        # Tokenize
        pred_tokens = predicted_text.lower().split()
        ref_tokens = reference_text.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
        
        # Calculate BLEU scores
        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
                bleu_scores[f'bleu{n}'] = score
            except:
                bleu_scores[f'bleu{n}'] = 0.0
        
        return bleu_scores
    
    def calculate_rouge(
        self,
        predicted_text: str,
        reference_text: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            predicted_text: Predicted text
            reference_text: Reference text
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not predicted_text or not reference_text:
            return {
                'rouge1_f': 0.0,
                'rouge2_f': 0.0,
                'rougeL_f': 0.0
            }
        
        scores = self.rouge_scorer.score(reference_text, predicted_text)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    
    def calculate_edit_distance(
        self,
        predicted_text: str,
        reference_text: str
    ) -> Tuple[int, float]:
        """
        Calculate Levenshtein edit distance and similarity ratio.
        
        Args:
            predicted_text: Predicted text
            reference_text: Reference text
            
        Returns:
            Tuple of (edit_distance, similarity_ratio)
        """
        matcher = SequenceMatcher(None, predicted_text, reference_text)
        similarity = matcher.ratio()
        
        # Approximate edit distance
        max_len = max(len(predicted_text), len(reference_text))
        edit_distance = int(max_len * (1 - similarity))
        
        return edit_distance, similarity
    
    def calculate_keyword_recovery(
        self,
        predicted_text: str,
        reference_text: str,
        min_word_length: int = 4
    ) -> Dict[str, float]:
        """
        Calculate keyword recovery rate.
        
        Args:
            predicted_text: Predicted text
            reference_text: Reference text
            min_word_length: Minimum word length to consider
            
        Returns:
            Dictionary with keyword recovery metrics
        """
        # Extract keywords (words longer than min_length)
        def extract_keywords(text):
            words = re.findall(r'\b\w+\b', text.lower())
            return set(w for w in words if len(w) >= min_word_length)
        
        pred_keywords = extract_keywords(predicted_text)
        ref_keywords = extract_keywords(reference_text)
        
        if len(ref_keywords) == 0:
            return {
                'keyword_precision': 0.0,
                'keyword_recall': 0.0,
                'keyword_f1': 0.0
            }
        
        # Calculate metrics
        intersection = pred_keywords & ref_keywords
        
        precision = len(intersection) / len(pred_keywords) if len(pred_keywords) > 0 else 0.0
        recall = len(intersection) / len(ref_keywords)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'keyword_precision': precision,
            'keyword_recall': recall,
            'keyword_f1': f1,
            'keywords_recovered': len(intersection),
            'total_keywords': len(ref_keywords)
        }
    
    def calculate_sensitive_info_leakage(
        self,
        predicted_text: str,
        reference_text: str,
        sensitive_patterns: List[str] = None
    ) -> Dict[str, any]:
        """
        Calculate leakage of sensitive information.
        
        Args:
            predicted_text: Predicted text
            reference_text: Reference text
            sensitive_patterns: List of regex patterns for sensitive info
            
        Returns:
            Dictionary with leakage metrics
        """
        if sensitive_patterns is None:
            # Default patterns for Brazilian legal documents
            sensitive_patterns = [
                r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b',  # CNPJ
                r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b',  # CPF
                r'\b[A-Z]{2}\s*\d{6}\b',  # OAB
                r'\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b',  # Process number
            ]
        
        leakage_results = {}
        
        for i, pattern in enumerate(sensitive_patterns):
            ref_matches = set(re.findall(pattern, reference_text))
            pred_matches = set(re.findall(pattern, predicted_text))
            
            leaked = ref_matches & pred_matches
            
            leakage_results[f'pattern_{i}_leaked'] = len(leaked)
            leakage_results[f'pattern_{i}_total'] = len(ref_matches)
            leakage_results[f'pattern_{i}_rate'] = (
                len(leaked) / len(ref_matches) if len(ref_matches) > 0 else 0.0
            )
        
        # Overall leakage rate
        total_sensitive = sum(v for k, v in leakage_results.items() if k.endswith('_total'))
        total_leaked = sum(v for k, v in leakage_results.items() if k.endswith('_leaked'))
        
        leakage_results['overall_leakage_rate'] = (
            total_leaked / total_sensitive if total_sensitive > 0 else 0.0
        )
        
        return leakage_results
    
    def evaluate_batch(
        self,
        predicted_texts: List[str],
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions.
        
        Args:
            predicted_texts: List of predicted texts
            reference_texts: List of reference texts
            
        Returns:
            Dictionary with averaged metrics
        """
        all_metrics = []
        
        for pred, ref in zip(predicted_texts, reference_texts):
            metrics = {}
            
            # BLEU scores
            metrics.update(self.calculate_bleu(pred, ref))
            
            # ROUGE scores
            metrics.update(self.calculate_rouge(pred, ref))
            
            # Edit distance
            edit_dist, similarity = self.calculate_edit_distance(pred, ref)
            metrics['edit_distance'] = edit_dist
            metrics['similarity'] = similarity
            
            # Keyword recovery
            metrics.update(self.calculate_keyword_recovery(pred, ref))
            
            # Sensitive info leakage
            metrics.update(self.calculate_sensitive_info_leakage(pred, ref))
            
            all_metrics.append(metrics)
        
        # Average all metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Pretty print metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("EMBEDDING INVERSION ATTACK EVALUATION")
        print("="*60)
        
        print("\n📊 Text Similarity Metrics:")
        print(f"  BLEU-1: {metrics.get('bleu1', 0):.4f}")
        print(f"  BLEU-2: {metrics.get('bleu2', 0):.4f}")
        print(f"  BLEU-4: {metrics.get('bleu4', 0):.4f}")
        print(f"  ROUGE-1: {metrics.get('rouge1_f', 0):.4f}")
        print(f"  ROUGE-2: {metrics.get('rouge2_f', 0):.4f}")
        print(f"  ROUGE-L: {metrics.get('rougeL_f', 0):.4f}")
        print(f"  Similarity: {metrics.get('similarity', 0):.4f}")
        
        print("\n🔑 Keyword Recovery:")
        print(f"  Precision: {metrics.get('keyword_precision', 0):.4f}")
        print(f"  Recall: {metrics.get('keyword_recall', 0):.4f}")
        print(f"  F1-Score: {metrics.get('keyword_f1', 0):.4f}")
        
        print("\n⚠️  Sensitive Information Leakage:")
        print(f"  Overall Leakage Rate: {metrics.get('overall_leakage_rate', 0):.4f}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Test metrics
    metrics = InversionMetrics()
    
    reference = "honorários assistenciais; honorários justiça trabalho; intimação / notificação"
    predicted = "honorários assistenciais; honorários justiça; notificação"
    
    print("Testing metrics...")
    print(f"\nReference: {reference}")
    print(f"Predicted: {predicted}")
    
    results = metrics.evaluate_batch([predicted], [reference])
    metrics.print_metrics(results)