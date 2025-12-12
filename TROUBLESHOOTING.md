# Troubleshooting Guide

## Common Issues and Solutions

### 1. ValueError: Array Dimensions Mismatch in Embedding Generation

**Error Message:**
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly, 
but along dimension 1, the array at index 0 has size 235 and the array at index 1 has size 157
```

**Cause:** 
This error occurred because texts have different lengths, resulting in attention masks with varying dimensions that cannot be stacked.

**Solution:** 
✅ **FIXED** - The code has been updated to only return embeddings (not attention masks), since we only need the [CLS] token embeddings for the inversion attack.

**Verification:**
```bash
python run_experiment.py --max_samples 100 --steps 1
```

---

### 2. CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Option 1: Reduce batch size
python run_experiment.py --batch_size 8

# Option 2: Use fewer samples
python run_experiment.py --max_samples 1000

# Option 3: Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python run_experiment.py
```

---

### 3. Module Not Found Errors

**Error Message:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### 4. Dataset File Not Found

**Error Message:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'updated_dataset_preprocessed.parquet_new.gzip'
```

**Solution:**
```bash
# Verify file exists
ls -lh updated_dataset_preprocessed.parquet_new.gzip

# Or specify full path
python run_experiment.py --data_path /full/path/to/dataset.gzip
```

---

### 5. Slow Training on CPU

**Symptom:** Training takes very long time

**Solutions:**
```bash
# Use smaller sample size
python run_experiment.py --max_samples 1000

# Use simpler model
python run_experiment.py --model_type mlp

# Reduce epochs
python run_experiment.py --epochs 10
```

---

### 6. Model File Not Found During Evaluation

**Error Message:**
```
FileNotFoundError: models/attacker/mlp/best_inverter.pt not found
```

**Solution:**
```bash
# Train the model first
python run_experiment.py --steps 2

# Or run all steps
python run_experiment.py --steps 1,2,3
```

---

### 7. Import Errors in Jupyter Notebook

**Error Message:**
```
ImportError: cannot import name 'LaborDataLoader'
```

**Solution:**
```python
# Add parent directory to path at the start of notebook
import sys
sys.path.append('..')
sys.path.append('../src')
```

---

### 8. Tokenizer Download Issues

**Error Message:**
```
OSError: Can't load tokenizer for 'neuralmind/bert-base-portuguese-cased'
```

**Solution:**
```bash
# Ensure internet connection is available
# Or download model manually:
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')"
```

---

### 9. Permission Denied When Creating Directories

**Error Message:**
```
PermissionError: [Errno 13] Permission denied: 'data/embeddings'
```

**Solution:**
```bash
# Create directories manually with proper permissions
mkdir -p data/embeddings models/attacker results/plots

# Or run with appropriate permissions
sudo python run_experiment.py  # Not recommended
# Better: fix directory permissions
chmod -R u+w data/ models/ results/
```

---

### 10. Pickle Protocol Error

**Error Message:**
```
ValueError: unsupported pickle protocol: 5
```

**Solution:**
```bash
# Upgrade Python to 3.8+
python --version  # Check version

# Or use compatible pickle protocol in code
# (Already handled in the implementation)
```

---

## Performance Optimization Tips

### For Faster Experimentation

```bash
# Minimal test run
python run_experiment.py \
  --max_samples 100 \
  --epochs 5 \
  --batch_size 16 \
  --model_type mlp
```

### For Production Results

```bash
# Full dataset with best model
python run_experiment.py \
  --max_samples 50000 \
  --epochs 30 \
  --batch_size 32 \
  --model_type attention
```

### Memory-Efficient Configuration

```bash
# For systems with limited RAM/VRAM
python run_experiment.py \
  --max_samples 5000 \
  --batch_size 8 \
  --model_type mlp
```

---

## Debugging Tips

### Enable Verbose Logging

```python
# Add to your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Monitor Memory Usage

```bash
# During training, monitor GPU memory
watch -n 1 nvidia-smi

# Or CPU/RAM usage
htop
```

### Test Individual Components

```bash
# Test data loading only
python -m src.preprocessing.data_loader

# Test embedding generation only
python -m src.embedding.bertimbau_embedder

# Test model architecture only
python -m src.attack.inverter_model
```

---

## Getting Help

If you encounter an issue not listed here:

1. Check the error message carefully
2. Verify all dependencies are installed: `pip list`
3. Ensure dataset file exists and is readable
4. Try with minimal configuration first
5. Check system resources (RAM, disk space, GPU memory)

For persistent issues, review the code documentation in each module.