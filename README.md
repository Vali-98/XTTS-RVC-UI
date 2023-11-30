# XTTS-RVC-UI

This is a simple UI that utilize's [Coqui's XTTSv2](https://github.com/coqui-ai/TTS) paired with RVC functionality to improve output quality.

# Usage

Simply clone this repository:

```
git clone https://github.com/Vali-98/XTTS-RVC-UI.git
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

