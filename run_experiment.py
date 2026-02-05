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
    print(f"Model types: MLP, LSTM, Attention, Categorical LSTM (all models)")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("="*80 + "\n")
    
    # Create directories
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('models/attacker', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Create subdirectories for each model
    for model_type in ['mlp', 'lstm', 'attention', 'categorical_lstm']:
        os.makedirs(f'models/attacker/{model_type}', exist_ok=True)
        os.makedirs(f'results/{model_type}', exist_ok=True)


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
    """Step 2: Train all inverter models (MLP, LSTM, Attention)."""
    print("\n" + "="*80)
    print("STEP 2: TRAINING ALL INVERTER MODELS")
    print("="*80 + "\n")
    
    model_types = ['mlp', 'lstm', 'attention', 'categorical_lstm']
    
    # Check if all models exist
    all_exist = all(os.path.exists(f'models/attacker/{mt}/best_inverter.pt') for mt in model_types)
    
    if all_exist and not args.force:
        print("✓ All models already exist. Use --force to retrain.")
        for mt in model_types:
            print(f"  - models/attacker/{mt}/best_inverter.pt")
        return
    
    # Import and run training (trains all models)
    from attack.train_inverter import main as train_main
    
    print("Training all four models: MLP, LSTM, Attention, and Categorical LSTM")
    print("This will train each model sequentially...\n")
    
    try:
        train_main()
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n✓ Step 2 completed successfully!")
    print("✓ All models trained:")
    for mt in model_types:
        if os.path.exists(f'models/attacker/{mt}/best_inverter.pt'):
            print(f"  ✓ {mt.upper()}: models/attacker/{mt}/best_inverter.pt")


def step3_evaluate_attack(args):
    """Step 3: Evaluate all attack models and generate comparison reports."""
    print("\n" + "="*80)
    print("STEP 3: EVALUATING ALL ATTACK MODELS")
    print("="*80 + "\n")
    
    from evaluation.evaluate_attack import evaluate_all_models
    
    # Check if embeddings exist
    if not os.path.exists('data/embeddings/test_embeddings.pkl'):
        print("✗ Test embeddings not found")
        print("Please run step 1 first to generate embeddings")
        return
    
    # Check which models exist
    model_types = ['mlp', 'lstm', 'attention', 'categorical_lstm']
    existing_models = [mt for mt in model_types if os.path.exists(f'models/attacker/{mt}/best_inverter.pt')]
    
    if not existing_models:
        print("✗ No trained models found")
        print("Please run training first (step 2)")
        return
    
    print(f"Found {len(existing_models)} trained model(s): {', '.join([m.upper() for m in existing_models])}")
    print("Evaluating all models and generating comparison reports...\n")
    
    try:
        # Evaluate all models
        all_results = evaluate_all_models(
            embeddings_path='data/embeddings/test_embeddings.pkl',
            output_dir='results',
            num_samples=args.eval_samples,
            num_examples=5
        )
        
        print("\n✓ Step 3 completed successfully!")
        print("✓ Generated reports:")
        for mt in existing_models:
            print(f"  - results/{mt}/ATTACK_REPORT.md")
            print(f"  - results/{mt}/attack_metrics.json")
            print(f"  - results/{mt}/reconstruction_examples.txt")
        if len(existing_models) > 1:
            print(f"  - results/reconstruction_examples_all_models.txt")
            print(f"  - results/MODEL_COMPARISON.md")
            
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


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
    
    # Training arguments (removed --model_type since we train all models)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training and inference'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
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
        print("\nData:")
        print("  - data/embeddings/train_embeddings.pkl")
        print("  - data/embeddings/test_embeddings.pkl")
        print("\nTrained Models:")
        for model_type in ['mlp', 'lstm', 'attention', 'categorical_lstm']:
            model_path = f"models/attacker/{model_type}/best_inverter.pt"
            if os.path.exists(model_path):
                print(f"  ✓ {model_type.upper()}: {model_path}")
        print("\nEvaluation Results:")
        print("  - results/training_summary.txt")
        if os.path.exists('results/reconstruction_examples_all_models.txt'):
            print("  - results/reconstruction_examples_all_models.txt")
        if os.path.exists('results/MODEL_COMPARISON.md'):
            print("  - results/MODEL_COMPARISON.md")
        print("\nIndividual Model Reports:")
        for model_type in ['mlp', 'lstm', 'attention', 'categorical_lstm']:
            if os.path.exists(f'results/{model_type}/ATTACK_REPORT.md'):
                print(f"  - results/{model_type}/ATTACK_REPORT.md")
                print(f"  - results/{model_type}/attack_metrics.json")
                print(f"  - results/{model_type}/reconstruction_examples.txt")
                print(f"  - results/{model_type}/plots/*.png")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()