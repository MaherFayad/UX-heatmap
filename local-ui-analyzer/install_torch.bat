@echo off
echo Uninstalling existing torch versions...
pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch Nightly with CUDA 12.8 support...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo.
echo Installation complete. Running verification...
python test_installation.py
pause
