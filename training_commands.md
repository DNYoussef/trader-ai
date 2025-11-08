# AI Training Commands Reference - Gary×Taleb Trading System

This document contains all the command-line instructions for training AI models outside of VSCode to avoid sudden shutdown issues.

## Prerequisites

Before running any training scripts, ensure:

1. **Python Dependencies Installed**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables** (if using Hugging Face models):
   Create a `.env` file with:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

3. **GPU Setup** (optional but recommended for faster training):
   - CUDA toolkit installed
   - PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Basic Training Commands

### 1. Simple ML Models Training
Quick training for basic models (Random Forest, Gradient Boosting, LSTM):
```bash
python scripts/training/simple_train.py
```
- **Duration**: 5-10 minutes
- **Output**: Creates models in `trained_models/` directory
- **Use case**: Quick testing and basic predictions

### 2. Execute Main Training Workflow
Complete training pipeline with all models:
```bash
python scripts/training/execute_training.py
```
- **Duration**: 30-60 minutes
- **Output**: Full model suite with registry metadata
- **Use case**: Production-ready models

## Advanced HRM (Hierarchical Reasoning Model) Training

### 3. Enhanced 32-Dimensional HRM Training
Full-scale training with TimesFM + FinGPT enhanced features:
```bash
python scripts/training/train_enhanced_hrm_32d.py
```
- **Duration**: 3-5 hours on GPU
- **Memory Required**: 8GB+ GPU RAM
- **Output**: `models/enhanced_hrm_32d_model.pth`
- **Features**: 32-dimensional input, grokking optimization

### 4. Optimized HRM Training (Faster)
Optimized version with torch.compile and mixed precision:
```bash
python scripts/training/train_optimized_hrm_32d.py
```
- **Duration**: 30-60 minutes on GPU (6-10x faster)
- **Memory Required**: 6GB+ GPU RAM
- **Output**: `models/optimized_hrm_32d_history.json`
- **Features**: Reduced model size, faster convergence

### 5. Resume Training from Checkpoint
Continue interrupted training:
```bash
python scripts/training/train_enhanced_hrm_32d_resume.py
```
- **Use case**: Resume after interruption
- **Requires**: Previous checkpoint files

## Monitoring and Testing

### 6. Live Training Monitor
Real-time monitoring of ongoing training:
```bash
python scripts/training/monitor_live_training.py
```
- **Use case**: Track training progress in separate terminal
- **Features**: Live metrics, loss curves, validation scores

### 7. Quick Data Test
Verify data pipeline before training:
```bash
python scripts/training/quick_data_test.py
```
- **Duration**: 1-2 minutes
- **Use case**: Validate data loading and preprocessing

### 8. Test Data Simple
Basic data validation:
```bash
python scripts/training/test_data_simple.py
```
- **Duration**: < 1 minute
- **Use case**: Quick sanity check

## Training in Background (Windows)

To run training in background and avoid VSCode shutdown issues:

### Option 1: Using Command Prompt
```bash
# Open new Command Prompt window
start cmd /k python scripts/training/train_enhanced_hrm_32d.py
```

### Option 2: Using PowerShell
```powershell
Start-Process python -ArgumentList "scripts/training/train_enhanced_hrm_32d.py" -NoNewWindow
```

### Option 3: Using Screen/Tmux (Git Bash/WSL)
```bash
# Start a new screen session
screen -S training

# Run training
python scripts/training/train_enhanced_hrm_32d.py

# Detach with Ctrl+A then D
# Reattach later with:
screen -r training
```

## Batch Training Script

For convenience, use the provided batch file:
```bash
start_training.bat
```

This will:
- Check dependencies
- Set up environment
- Launch training in a new window
- Monitor progress

## Training Output Locations

- **Models**: `models/` and `trained_models/`
- **Checkpoints**: `models/checkpoints/`
- **Logs**: `logs/training/`
- **Metrics**: `models/training_metrics_live.json`
- **Visualizations**: `models/*.png`, `models/*.pdf`

## Common Issues and Solutions

### Out of Memory Error
- Reduce batch size in training script
- Use `train_optimized_hrm_32d.py` instead
- Close other applications

### Module Not Found
```bash
pip install -r requirements.txt
pip install -r src/intelligence/requirements.txt
```

### CUDA Not Available
- Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA toolkit
- Reinstall PyTorch with CUDA support

### Training Interrupted
- Use resume script: `train_enhanced_hrm_32d_resume.py`
- Check `models/checkpoints/` for latest checkpoint

## Performance Tips

1. **Close unnecessary applications** to free RAM/VRAM
2. **Use GPU** for 10-100x speedup
3. **Run optimized versions** for faster training
4. **Monitor with separate terminal** to track progress
5. **Save checkpoints frequently** (already configured in scripts)

## Production Deployment

After training completes:

1. **Verify models**:
   ```bash
   python scripts/verify_model.py
   ```

2. **Test in paper trading**:
   ```bash
   python main.py --mode paper
   ```

3. **Deploy to production**:
   ```bash
   python scripts/deployment/launch_enhanced_paper_trading.py
   ```

## Support

For issues or questions:
- Check logs in `logs/training/`
- Review error messages in terminal
- Verify all dependencies installed
- Ensure sufficient system resources

---

**Note**: Always test models thoroughly in paper trading before using with real capital.