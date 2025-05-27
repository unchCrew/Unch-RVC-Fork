import json
import os
import random
import sys

import gradio as gr

now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from core import run_tts_script
from tabs.inference.inference import (
    change_choices,
    get_indexes,
    get_speakers_id,
    match_index,
    names,
    default_weight,
)

i18n = I18nAuto()

with open(
    os.path.join("rvc", "lib", "tools", "tts_voices.json"), "r", encoding="utf-8"
) as file:
    tts_voices_data = json.load(file)

short_names = [voice.get("ShortName", "") for voice in tts_voices_data]

def process_input(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file.read()
        gr.Info(f"The file has been loaded!")
        return file_path, file_path
    except UnicodeDecodeError:
        gr.Info(f"The file has to be in UTF-8 encoding.")
        return None, None

# TTS tab
def tts_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model for conversion."),
                choices=sorted(names),
                interactive=True,
                value=default_weight,
            )
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the index file for conversion."),
                choices=get_indexes(),
                value=match_index(model_file.value),
                interactive=True,
            )
        with gr.Row():
            unload_button = gr.Button(i18n("Unload Voice"))
            refresh_button = gr.Button(i18n("Refresh"))

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file],
            )

            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    gr.Markdown(
        i18n(
            "Applio uses EdgeTTS for Text-to-Speech conversion. Learn more [here](https://docs.applio.org/applio/getting-started/tts)."
        )
    )
    tts_voice = gr.Dropdown(
        label=i18n("TTS Voice"),
        info=i18n("Select the TTS voice."),
        choices=short_names,
        interactive=True,
        value=random.choice(short_names),
    )

    tts_rate = gr.Slider(
        minimum=-50,
        maximum=50,
        step=1,
        label=i18n("TTS Speed"),
        info=i18n("Adjust the speed of the TTS voice."),
        value=0,
        interactive=True,
    )

    with gr.Tabs():
        with gr.Tab(label="Text to Speech"):
            tts_text = gr.Textbox(
                label=i18n("Text to Synthesize"),
                placeholder=i18n("Enter text to synthesize"),
                lines=3,
                interactive=True,
            )
        with gr.Tab(label="File to Speech"):
            txt_file = gr.File(
                label=i18n("Upload .txt file"),
                type="filepath",
            )
            input_tts_path = gr.Textbox(
                label=i18n("Text File Path"),
                placeholder=i18n("Path to text file for TTS."),
                interactive=True,
            )

    with gr.Accordion(i18n("Settings"), open=False):
        output_tts_path = gr.Textbox(
            label=i18n("Output TTS Audio"),
            placeholder=i18n("Enter output path"),
            value=os.path.join(now_dir, "assets", "audios", "tts_output.wav"),
            interactive=True,
        )
        output_rvc_path = gr.Textbox(
            label=i18n("Output RVC Audio"),
            placeholder=i18n("Enter output path"),
            value=os.path.join(now_dir, "assets", "audios", "tts_rvc_output.wav"),
            interactive=True,
        )
        export_format = gr.Radio(
            label=i18n("Export Format"),
            choices=["WAV", "MP3"],
            value="WAV",
            interactive=True,
        )
        sid = gr.Dropdown(
            label=i18n("Speaker ID"),
            choices=get_speakers_id(model_file.value),
            value=0,
            interactive=True,
        )
        pitch = gr.Slider(
            minimum=-12,
            maximum=12,
            step=1,
            label=i18n("Pitch"),
            info=i18n("Adjust the pitch of the audio."),
            value=0,
            interactive=True,
        )

    terms_checkbox = gr.Checkbox(
        label=i18n("I agree to the terms of use"),
        info=i18n("See [terms](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md)."),
        value=False,
        interactive=True,
    )
    convert_button = gr.Button(i18n("Convert"))

    with gr.Row():
        vc_output1 = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("Conversion details will appear here."),
        )
        vc_output2 = gr.Audio(label=i18n("Output Audio"))

    def enforce_terms(terms_accepted, *args):
        if not terms_accepted:
            gr.Info("You must agree to the Terms of Use to proceed.")
            return "Terms not accepted", None
        return run_tts_script(*args)

    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, sid, sid],
    )
    txt_file.upload(
        fn=process_input,
        inputs=[txt_file],
        outputs=[input_tts_path, txt_file],
    )
    convert_button.click(
        fn=enforce_terms,
        inputs=[
            terms_checkbox,
            input_tts_path,
            tts_text,
            tts_voice,
            tts_rate,
            pitch,
            output_tts_path,
            output_rvc_path,
            model_file,
            index_file,
            export_format,
            sid,
        ],
        outputs=[vc_output1, vc_output2],
    )
