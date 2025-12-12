"""
Main script to run the complete embedding inversion experiment.
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from preprocessing.data_loader import LaborDataLoader
from embedding.bertimbau_embedder import generate_embeddings_from_data
from attack.train_inverter import main as train_inverter
from evaluation.evaluate_attack import main as evaluate_attack


def setup_experiment(args):
    """Setup experiment directories and configuration."""
    print("="*80)
    print("EMBEDDING INVERSION ATTACK EXPERIMENT")
    print("Análise de Riscos de Reconstrução Textual em Modelos de Predição")
    print("="*80)
    print(f"\nExperiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {args.data_path}")
    print(f"Max samples: {args.max_samples}")
    print(f"Model type: {args.model_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("="*80 + "\n")
    
    # Create directories
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('models/attacker', exist_ok=True)
    os.makedirs('results', exist_ok=True)


def step1_generate_embeddings(args):
    """Step 1: Generate embeddings from dataset."""
    print("\n" + "="*80)
    print("STEP 1: GENERATING EMBEDDINGS")
    print("="*80 + "\n")
    
    if os.path.exists('data/embeddings/train_embeddings.pkl') and not args.force:
        print("✓ Embeddings already exist. Use --force to regenerate.")
        return
    
    generate_embeddings_from_data(
        data_path=args.data_path,
        output_dir='data/embeddings',
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    print("\n✓ Step 1 completed successfully!")


def step2_train_inverter(args):
    """Step 2: Train the inverter model."""
    print("\n" + "="*80)
    print("STEP 2: TRAINING INVERTER MODEL")
    print("="*80 + "\n")
    
    model_path = f'models/attacker/{args.model_type}/best_inverter.pt'
    if os.path.exists(model_path) and not args.force:
        print(f"✓ Model already exists at {model_path}. Use --force to retrain.")
        return
    
    # Import and run training
    from attack.train_inverter import main as train_main
    
    # Temporarily modify sys.argv for the training script
    original_argv = sys.argv
    sys.argv = [
        'train_inverter.py',
        '--model_type', args.model_type,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--learning_rate', str(args.learning_rate)
    ]
    
    try:
        train_main()
    finally:
        sys.argv = original_argv
    
    print("\n✓ Step 2 completed successfully!")


def step3_evaluate_attack(args):
    """Step 3: Evaluate the attack and generate report."""
    print("\n" + "="*80)
    print("STEP 3: EVALUATING ATTACK")
    print("="*80 + "\n")
    
    from evaluation.evaluate_attack import AttackEvaluator
    
    model_path = f'models/attacker/{args.model_type}/best_inverter.pt'
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        print("Please run training first (step 2)")
        return
    
    # Create evaluator
    evaluator = AttackEvaluator(
        model_path=model_path,
        model_type=args.model_type
    )
    
    # Evaluate
    results = evaluator.evaluate_on_dataset(
        embeddings_path='data/embeddings/test_embeddings.pkl',
        num_samples=args.eval_samples
    )
    
    # Generate report
    evaluator.generate_report(results, output_dir='results')
    
    print("\n✓ Step 3 completed successfully!")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description='Run embedding inversion attack experiment'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='updated_dataset_preprocessed.parquet_new.gzip',
        help='Path to input dataset'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=5000,
        help='Maximum number of samples to use (None for all)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_type',
        type=str,
        default='mlp',
        choices=['mlp', 'lstm', 'attention'],
        help='Type of inverter model'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training and inference'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--eval_samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (None for all)'
    )
    
    # Execution control
    parser.add_argument(
        '--steps',
        type=str,
        default='1,2,3',
        help='Steps to run (comma-separated: 1,2,3)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration of existing files'
    )
    
    args = parser.parse_args()
    
    # Parse steps
    steps = [int(s.strip()) for s in args.steps.split(',')]
    
    # Setup
    setup_experiment(args)
    
    try:
        # Run requested steps
        if 1 in steps:
            step1_generate_embeddings(args)
        
        if 2 in steps:
            step2_train_inverter(args)
        
        if 3 in steps:
            step3_evaluate_attack(args)
        
        # Final summary
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nGenerated files:")
        print("  - data/embeddings/train_embeddings.pkl")
        print("  - data/embeddings/test_embeddings.pkl")
        print(f"  - models/attacker/{args.model_type}/best_inverter.pt")
        print("  - results/attack_metrics.json")
        print("  - results/ATTACK_REPORT.md")
        print("  - results/reconstruction_examples.txt")
        print("  - results/plots/*.png")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()