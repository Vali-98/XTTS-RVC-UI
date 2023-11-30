import torch
from TTS.api import TTS
import gradio as gr
from rvc import Config, load_hubert, get_vc, rvc_infer
import gc , os, requests
from pathlib import Path

def download_model():
    if(not os.path.isfile('./models/xtts/model.pth')):
       print('Downloading model.pth')
       r = requests.get('https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/model.pth')
       with open('./models/xtts/model.pth', 'wb') as f:
            f.write(r.content)

    if(not os.path.isfile('./models/xtts/vocab.json')):
       print('Downloading vocab.json')
       r = requests.get('https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/vocab.json')
       with open('./models/xtts/vocab.json', 'wb') as f:
            f.write(r.content)

    if(not os.path.isfile('./models/xtts/config.json')):
       print('Downloading config.json')
       r = requests.get('https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/config.json')
       with open('./models/xtts/config.json', 'wb') as f:
            f.write(r.content)
            
    if(not os.path.isfile('./models/xtts/dvae.pth')):
       print('Downloading dvae.pth')
       r = requests.get('https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/dvae.pth')
       with open('./models/xtts/dvae.pth', 'wb') as f:
            f.write(r.content)
            
    if(not os.path.isfile('./models/xtts/mel_stats.pth')):
       print('Downloading mel_stats.pth')
       r = requests.get('https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/mel_stats.pth')
       with open('./models/xtts/mel_stats.pth', 'wb') as f:
            f.write(r.content)

[Path(_dir).mkdir(parents=True, exist_ok=True) for _dir in ['./models/xtts', './voices', './rvcs']]

#download_model()

device_tts = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: " + device_tts) 
tts = TTS(model_path="./models/xtts", config_path='./models/xtts/config.json').to(device_tts)
device = "cuda:0"

voices = []
rvcs = []
        
def get_dirs():
    global voices 
    voices = os.listdir("./voices")
    global rvcs
    rvcs = list(filter(lambda x:x.endswith(".pth"), os.listdir("./rvcs")))

def runtts(rvc, voice, text): 
    audio = tts.tts_to_file(text=text, speaker_wav="./voices/" + voice, language="en", file_path="./output.wav")
    voice_change(rvc)
    return ["./output.wav" , "./outputrvc.wav"]
    
def main():
    get_dirs()
    print(rvcs)
    print(voices)
    text_input = gr.Textbox(placeholder="Write here...")
    audio_output = gr.Audio(label="TTS result", type="filepath")
    rvc_audio_output = gr.Audio(label="RVC result", type="filepath")
    rvc_dropdown = gr.Dropdown(choices=rvcs, value=rvcs[0]) 
    voice_dropdown = gr.Dropdown(choices=voices, value=voices[0])
    
    interface= gr.Interface(fn=runtts, inputs=[rvc_dropdown, voice_dropdown, text_input], outputs=[audio_output, rvc_audio_output], title="Coqui TTS UI")
    interface.launch(server_name="0.0.0.0", server_port=5000, quiet=True)

config = Config(device, True)
hubert_model = load_hubert("cuda", config.is_half, "./models/hubert_base.pt")

def voice_change(rvc):
    modelname = os.path.splitext(rvc)[0]
    print(modelname)
    rvc_model_path = "./rvcs/" + rvc  
    rvc_index_path = "./rvcs/" + modelname + ".index" if os.path.isfile("./rvc/" + modelname + ".index") else ""
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, "./rvcs/" + rvc)
    rvc_infer(rvc_index_path, 0.2, "./output.wav", "./outputrvc.wav", 0, "rmvpe", cpt, version, net_g, 3, tgt_sr, 0.25, 0, 0, vc, hubert_model)
    gc.collect()
    
if __name__ == "__main__":
    main()
