import torch
from TTS.api import TTS
import gradio as gr
from rvc import Config, load_hubert, get_vc, rvc_infer
import gc , os, requests
from pathlib import Path

def download_models():
	rvc_files = ['hubert_base.pt', 'rmvpe.pt']

	for file in rvc_files: 
		if(not os.path.isfile(f'./models/{file}')):
			print('Downloading rmvpe.pt')
			r = requests.get(f'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/{file}')
			with open(f'./models/{file}', 'wb') as f:
					f.write(r.content)

	xtts_files = ['vocab.json', 'config.json', 'dvae.path', 'mel_stats.pth', 'model.pth']

	for file in xtts_files:
		if(not os.path.isfile(f'./models/xtts/{file}')):
			print('Downloading model.pth')
			r = requests.get(f'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/{file}')
			with open(f'./models/xtts/{file}', 'wb') as f:
				f.write(r.content)

download_models()

[Path(_dir).mkdir(parents=True, exist_ok=True) for _dir in ['./models/xtts', './voices', './rvcs']]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: " + device) 

config = Config(device, device != 'cpu')
hubert_model = load_hubert(device, config.is_half, "./models/hubert_base.pt")
tts = TTS(model_path="./models/xtts", config_path='./models/xtts/config.json').to(device)
voices = []
rvcs = []
        
def get_rvc_voices():
	global voices 
	voices = os.listdir("./voices")
	global rvcs
	rvcs = list(filter(lambda x:x.endswith(".pth"), os.listdir("./rvcs")))
	return [rvcs, voices]

def runtts(rvc, voice, text): 
    audio = tts.tts_to_file(text=text, speaker_wav="./voices/" + voice, language="en", file_path="./output.wav")
    voice_change(rvc)
    return ["./output.wav" , "./outputrvc.wav"]
    
def main():
	get_rvc_voices()
	print(rvcs)
	print(voices)
	with gr.Blocks(title='TTS RVC UI') as interface:
		with gr.Row():
			gr.Markdown("""
				#XTTS RVC UI
			""")
		with gr.Row(): 
			with gr.Column():
				rvc_dropdown = gr.Dropdown(choices=rvcs, value=rvcs[0] if len(rvcs) > 0 else '', label='RVC model') 
				voice_dropdown = gr.Dropdown(choices=voices, value=voices[0] if len(voices) > 0 else '', label='Voice sample')
				refresh_button = gr.Button(value='Refresh')
				text_input = gr.Textbox(placeholder="Write here...")
				submit_button = gr.Button(value='Submit')
			with gr.Column():        
				audio_output = gr.Audio(label="TTS result", type="filepath", interactive=False)
				rvc_audio_output = gr.Audio(label="RVC result", type="filepath", interactive=False)

		submit_button.click(inputs=[rvc_dropdown, voice_dropdown, text_input], outputs=[audio_output, rvc_audio_output], fn=runtts)
		def refresh_dropdowns():
			get_rvc_voices()
			print('Refreshed voice and RVC list!')
			return [gr.update(choices=rvcs, value=rvcs[0] if len(rvcs) > 0 else ''),  gr.update(choices=voices, value=voices[0] if len(voices) > 0 else '')] 

		refresh_button.click(fn=refresh_dropdowns, outputs=[rvc_dropdown, voice_dropdown])

	interface.launch(server_name="0.0.0.0", server_port=5000, quiet=True)

def voice_change(rvc):
    modelname = os.path.splitext(rvc)[0]
    print(modelname)
    rvc_model_path = "./rvcs/" + rvc  
    rvc_index_path = "./rvcs/" + modelname + ".index" if os.path.isfile("./rvc/" + modelname + ".index") else ""
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, "./rvcs/" + rvc)
    rvc_infer(rvc_index_path, 0.2, "./output.wav", "./outputrvc.wav", 0, "rmvpe", cpt, version, net_g, 3, tgt_sr, 0.25, 0, 0, vc, hubert_model)
    del cpt
    gc.collect()
    
if __name__ == "__main__":
    main()
