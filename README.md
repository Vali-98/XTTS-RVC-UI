# XTTS-RVC-UI

This is a simple UI that utilize's [Coqui's XTTSv2](https://github.com/coqui-ai/TTS) paired with RVC functionality to improve output quality.

# Prerequisites

- Requires MSVC - VC 2022 C++ x64/x86 build tools.

# Usage

Clone this repository:

```
git clone https://github.com/Vali-98/XTTS-RVC-UI.git
```

It is recommended to create a venv.

Then, install the requirements:

```
pip install -r requirements.txt
```

If you have a CUDA device available, it is also recommended to install PyTorch with CUDA for faster conversions.

```
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Then run `start.bat` , `start.sh` or simply `python app.py`

This will create the following folders within the project:

```
\models\xtts
\rvcs
\voices
```
- Relevant models will be downloaded into `\models`. This will be approximately ~2.27GB.
- You can manually add the desired XTTSv2 model files in `\models\xtts`.
- Place RVC models in `\rvcs`. Rename them as needed. If an **identically named** .index file exists in `\rvcs`, it will also be used.
- Place voice samples in `\voices`

