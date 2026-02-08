# Setup Without Conda

Since conda is not installed, follow these steps to set up using Python's built-in `venv` module:

## Quick Start

1. **Create and activate the main environment:**
   ```bash
   ./setup_venv.sh
   source venv/bin/activate
   ```

2. **For STT service (if needed):**
   ```bash
   ./setup_stt_env_venv.sh
   ```

3. **Download the STT model:**
   ```bash
   python -m konkani_agent.utils.model_download
   ```

4. **Set your Gemini API key:**
   ```bash
   export GEMINI_API_KEY=your_actual_key_here
   ```

5. **Run the pipeline:**
   ```bash
   ./start_pipeline.sh
   ```

## What Changed

- **Removed conda dependency** - Both `start_pipeline.sh` and `start_stt_service.sh` now use Python virtual environments (`venv`) instead
- **New setup scripts:**
  - `setup_venv.sh` - Creates main venv and installs all dependencies
  - `setup_stt_env_venv.sh` - Creates separate STT venv (optional, for NeMo isolation)
- **Fixed hardcoded paths** - PYTHONPATH now uses the correct project directory

## Troubleshooting

If you see "command not found" errors:
- Make sure you've run `./setup_venv.sh` first
- Activate the environment: `source venv/bin/activate`
- Check the environment is active (you should see `(venv)` in your terminal prompt)

If you need to start fresh:
```bash
rm -rf venv venv-stt
./setup_venv.sh
source venv/bin/activate
```
