import torch
from TTS.api import TTS
import gradio as gr
from rvc import Config, load_hubert, get_vc, rvc_infer
import gc , os, sys, argparse, requests
from pathlib import Path

parser = argparse.ArgumentParser(
	prog='XTTS-RVC-UI',
	description='Gradio UI for XTTSv2 and RVC'
)

parser.add_argument('-s', '--silent', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

if args.silent: 
	print('Enabling silent mode.')
	sys.stdout = open(os.devnull, 'w')

def download_models():
	rvc_files = ['hubert_base.pt', 'rmvpe.pt']

	for file in rvc_files: 
		if(not os.path.isfile(f'./models/{file}')):
			print(f'Downloading{file}')
			r = requests.get(f'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/{file}')
			with open(f'./models/{file}', 'wb') as f:
					f.write(r.content)

	xtts_files = ['vocab.json', 'config.json', 'dvae.path', 'mel_stats.pth', 'model.pth']

	for file in xtts_files:
		if(not os.path.isfile(f'./models/xtts/{file}')):
			print(f'Downloading {file}')
			r = requests.get(f'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/{file}')
			with open(f'./models/xtts/{file}', 'wb') as f:
				f.write(r.content)
				

[Path(_dir).mkdir(parents=True, exist_ok=True) for _dir in ['./models/xtts', './voices', './rvcs']]

download_models()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: " + device) 

config = Config(device, device != 'cpu')
hubert_model = load_hubert(device, config.is_half, "./models/hubert_base.pt")
tts = TTS(model_path="./models/xtts", config_path='./models/xtts/config.json').to(device)
voices = []
rvcs = []
langs = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]

def get_rvc_voices():
	global voices 
	voices = os.listdir("./voices")
	global rvcs
	rvcs = list(filter(lambda x:x.endswith(".pth"), os.listdir("./rvcs")))
	return [rvcs, voices]

def runtts(rvc, voice, text, pitch_change, index_rate, language): 
    audio = tts.tts_to_file(text=text, speaker_wav="./voices/" + voice, language=language, file_path="./output.wav")
    voice_change(rvc, pitch_change, index_rate)
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
				lang_dropdown = gr.Dropdown(choices=langs, value=langs[0], label='Language')
				rvc_dropdown = gr.Dropdown(choices=rvcs, value=rvcs[0] if len(rvcs) > 0 else '', label='RVC model') 
				voice_dropdown = gr.Dropdown(choices=voices, value=voices[0] if len(voices) > 0 else '', label='Voice sample')
				refresh_button = gr.Button(value='Refresh')
				text_input = gr.Textbox(placeholder="Write here...")
				submit_button = gr.Button(value='Submit')
				with gr.Row():
					pitch_slider = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="Pitch")
					index_rate_slider = gr.Slider(minimum=0, maximum=1, value=0.75, step=0.05, label="Index Rate")
			with gr.Column():        
				audio_output = gr.Audio(label="TTS result", type="filepath", interactive=False)
				rvc_audio_output = gr.Audio(label="RVC result", type="filepath", interactive=False)

		submit_button.click(inputs=[rvc_dropdown, voice_dropdown, text_input, pitch_slider, index_rate_slider, lang_dropdown], outputs=[audio_output, rvc_audio_output], fn=runtts)
		def refresh_dropdowns():
			get_rvc_voices()
			print('Refreshed voice and RVC list!')
			return [gr.update(choices=rvcs, value=rvcs[0] if len(rvcs) > 0 else ''),  gr.update(choices=voices, value=voices[0] if len(voices) > 0 else '')] 

		refresh_button.click(fn=refresh_dropdowns, outputs=[rvc_dropdown, voice_dropdown])

	interface.launch(server_name="0.0.0.0", server_port=5000, quiet=True)

# delete later

class RVC_Data:
	def __init__(self):
		self.current_model = {}
		self.cpt = {}
		self.version = {}
		self.net_g = {} 
		self.tgt_sr = {}
		self.vc = {} 

	def load_cpt(self, modelname, rvc_model_path):
		if self.current_model != modelname:
				print("Loading new model")
				del self.cpt, self.version, self.net_g, self.tgt_sr, self.vc
				self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(device, config.is_half, config, rvc_model_path)
				self.current_model = modelname

rvc_data = RVC_Data()

def voice_change(rvc, pitch_change, index_rate):
	modelname = os.path.splitext(rvc)[0]
	print("Using RVC model: "+ modelname)
	rvc_model_path = "./rvcs/" + rvc  
	rvc_index_path = "./rvcs/" + modelname + ".index" if os.path.isfile("./rvcs/" + modelname + ".index") and index_rate != 0 else ""

	if rvc_index_path != "" :
		print("Index file found!")

	#load_cpt(modelname, rvc_model_path)
	#cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)
	rvc_data.load_cpt(modelname, rvc_model_path)
	
	rvc_infer(
		index_path=rvc_index_path, 
		index_rate=index_rate, 
		input_path="./output.wav", 
		output_path="./outputrvc.wav", 
		pitch_change=pitch_change, 
		f0_method="rmvpe", 
		cpt=rvc_data.cpt, 
		version=rvc_data.version, 
		net_g=rvc_data.net_g, 
		filter_radius=3, 
		tgt_sr=rvc_data.tgt_sr, 
		rms_mix_rate=0.25, 
		protect=0, 
		crepe_hop_length=0, 
		vc=rvc_data.vc, 
		hubert_model=hubert_model
	)
	gc.collect()
    
if __name__ == "__main__":
    main()
