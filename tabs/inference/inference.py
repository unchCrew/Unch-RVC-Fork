import os, sys
import gradio as gr
import regex as re
import shutil
import datetime
import json
import torch

from core import (
    run_infer_script,
    run_batch_infer_script,
)

from assets.i18n.i18n import I18nAuto

from rvc.lib.utils import format_title
from tabs.settings.sections.restart import stop_infer

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")
custom_embedder_root = os.path.join(now_dir, "rvc", "models", "embedders", "embedders_custom")

PRESETS_DIR = os.path.join(now_dir, "assets", "presets")
FORMANTSHIFT_DIR = os.path.join(now_dir, "assets", "formant_shift")

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {"wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"}

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if file.endswith((".pth", ".onnx")) and not (file.startswith("G_") or file.startswith("D_"))
]

default_weight = names[0] if names else None

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext)) and root == audio_root_relative and "_output" not in name
]

custom_embedders = [
    os.path.join(dirpath, dirname)
    for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
    for dirname in dirnames
]

def update_sliders(preset):
    with open(os.path.join(PRESETS_DIR, f"{preset}.json"), "r", encoding="utf-8") as json_file:
        values = json.load(json_file)
    return values["pitch"], values["index_rate"], values["rms_mix_rate"], values["protect"]

def update_sliders_formant(preset):
    with open(os.path.join(FORMANTSHIFT_DIR, f"{preset}.json"), "r", encoding="utf-8") as json_file:
        values = json.load(json_file)
    return values["formant_qfrency"], values["formant_timbre"]

def export_presets_button(preset_name, pitch, index_rate, rms_mix_rate, protect):
    if preset_name:
        file_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        presets_data = {
            "pitch": pitch,
            "index_rate": index_rate,
            "rms_mix_rate": rms_mix_rate,
            "protect": protect,
        }
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(presets_data, json_file, ensure_ascii=False, indent=4)
        return "Preset saved successfully!"
    return "Please enter a preset name."

def import_presets_button(file_path):
    if file_path:
        with open(file_path.name, "r", encoding="utf-8") as json_file:
            presets = json.load(json_file)
        return list(presets.keys()), presets, "Presets imported successfully!"
    return [], {}, "No file selected for import."

def list_json_files(directory):
    return [f.rsplit(".", 1)[0] for f in os.listdir(directory) if f.endswith(".json")]

def refresh_presets():
    return gr.update(choices=list_json_files(PRESETS_DIR))

def output_path_fn(input_audio_path):
    original_name = os.path.basename(input_audio_path).rsplit(".", 1)[0]
    new_name = original_name + "_output.wav"
    return os.path.join(os.path.dirname(input_audio_path), new_name)

def change_choices(model):
    speakers = get_speakers_id(model) if model else [0]
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if file.endswith((".pth", ".onnx")) and not (file.startswith("G_") or file.startswith("D_"))
    ]
    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]
    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext)) and root == audio_root_relative and "_output" not in name
    ]
    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
        {"choices": sorted(speakers) if isinstance(speakers, (list, tuple)) else [0], "__type__": "update"},
        {"choices": sorted(speakers) if isinstance(speakers, (list, tuple)) else [0], "__type__": "update"},
    )

def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]
    return indexes_list if indexes_list else ""

def save_to_wav(record_button):
    if record_button:
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(audio_root_relative, new_name)
        shutil.move(record_button, target_path)
        return target_path, output_path_fn(target_path)
    return None, None

def save_to_wav2(upload_audio):
    if upload_audio:
        formated_name = format_title(os.path.basename(upload_audio))
        target_path = os.path.join(audio_root_relative, formated_name)
        if os.path.exists(target_path):
            os.remove(target_path)
        shutil.copy(upload_audio, target_path)
        return target_path, output_path_fn(target_path)
    return None, None

def delete_outputs():
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and "_output" in name:
                os.remove(os.path.join(root, name))
    return "Output audios cleared!"

def match_index(model_file_value):
    if model_file_value:
        model_folder = os.path.dirname(model_file_value)
        model_name = os.path.basename(model_file_value)
        index_files = get_indexes()
        pattern = r"^(.*?)_"
        match = re.match(pattern, model_name)
        for index_file in index_files:
            if os.path.dirname(index_file) == model_folder or (match and match.group(1) in os.path.basename(index_file)) or model_name in os.path.basename(index_file):
                return index_file
    return ""

def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."
    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)
    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)
    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name."
    os.makedirs(target_folder, exist_ok=True)
    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))
    if config_file:
        shutil.copy(config_file, os.path.join(target_folder, os.path.basename(config_file)))
    return f"Files moved to {target_folder}"

def refresh_formant():
    return gr.update(choices=list_json_files(FORMANTSHIFT_DIR))

def refresh_embedders_folders():
    return [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]

def get_speakers_id(model):
    if model:
        try:
            model_data = torch.load(os.path.join(now_dir, model), map_location="cpu", weights_only=True)
            speakers_id = model_data.get("speakers_id")
            return list(range(speakers_id)) if speakers_id else [0]
        except Exception:
            return [0]
    return [0]

# Inference tab
def inference_tab():
    with gr.Column():
        with gr.Row(equal_height=True):
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                choices=sorted(names),
                value=default_weight,
                interactive=True,
                allow_custom_value=True,
            )
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                choices=get_indexes(),
                value=match_index(default_weight) if default_weight else "",
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row(equal_height=True):
            unload_button = gr.Button(i18n("Unload Voice"))
            refresh_button = gr.Button(i18n("Refresh"))
            unload_button.click(
                fn=lambda: ({"value": "", "__type__": "update"}, {"value": "", "__type__": "update"}),
                inputs=[],
                outputs=[model_file, index_file],
            )
            model_file.select(
                fn=match_index,
                inputs=[model_file],
                outputs=[index_file],
            )

    # Single inference tab
    with gr.Tab(i18n("Single")):
        with gr.Column():
            upload_audio = gr.Audio(label=i18n("Upload Audio"), type="filepath")
            audio = gr.Dropdown(
                label=i18n("Select Audio"),
                choices=sorted(audio_paths),
                value=audio_paths[0] if audio_paths else "",
                interactive=True,
                allow_custom_value=True,
            )
            output_path = gr.Textbox(
                label=i18n("Output Path"),
                value=output_path_fn(audio_paths[0]) if audio_paths else os.path.join(audio_root, "output.wav"),
                interactive=True,
            )
            export_format = gr.Radio(
                label=i18n("Export Format"),
                choices=["WAV", "MP3", "FLAC"],
                value="WAV",
                interactive=True,
            )
            sid = gr.Dropdown(
                label=i18n("Speaker ID"),
                choices=get_speakers_id(model_file.value),
                value=0,
                interactive=True,
            )
            with gr.Accordion(i18n("Settings"), open=False):
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=i18n("Pitch"),
                    value=0,
                    interactive=True,
                )
                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Search Feature Ratio"),
                    value=0.75,
                    interactive=True,
                )
                rms_mix_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Volume Envelope"),
                    value=1,
                    interactive=True,
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protect Voiceless Consonants"),
                    value=0.5,
                    interactive=True,
                )
                split_audio = gr.Checkbox(
                    label=i18n("Split Audio"),
                    value=False,
                    interactive=True,
                )
                autotune = gr.Checkbox(
                    label=i18n("Autotune"),
                    value=False,
                    interactive=True,
                )
                autotune_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Autotune Strength"),
                    value=1,
                    interactive=True,
                    visible=False,
                )
                clean_audio = gr.Checkbox(
                    label=i18n("Clean Audio"),
                    value=False,
                    interactive=True,
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Clean Strength"),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                formant_shifting = gr.Checkbox(
                    label=i18n("Formant Shifting"),
                    value=False,
                    interactive=True,
                )
                formant_qfrency = gr.Slider(
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    label=i18n("Quefrency"),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                formant_timbre = gr.Slider(
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    label=i18n("Timbre"),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                f0_method = gr.Radio(
                    label=i18n("Pitch Extraction Algorithm"),
                    choices=["crepe", "rmvpe", "fcpe"],
                    value="rmvpe",
                    interactive=True,
                )
                hop_length = gr.Slider(
                    minimum=1,
                    maximum=512,
                    step=1,
                    label=i18n("Hop Length"),
                    value=128,
                    interactive=True,
                    visible=False,
                )
                embedder_model = gr.Radio(
                    label=i18n("Embedder Model"),
                    choices=["contentvec", "chinese-hubert-base", "japanese-hubert-base", "custom"],
                    value="contentvec",
                    interactive=True,
                )
                with gr.Column(visible=False) as embedder_custom:
                    embedder_model_custom = gr.Dropdown(
                        label=i18n("Custom Embedder"),
                        choices=refresh_embedders_folders(),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    refresh_embedders_button = gr.Button(i18n("Refresh Embedders"))
                    folder_name_input = gr.Textbox(label=i18n("Folder Name"))
                    bin_file_upload = gr.File(label=i18n("Upload .bin"), type="filepath")
                    config_file_upload = gr.File(label=i18n("Upload .json"), type="filepath")
                    move_files_button = gr.Button(i18n("Move Files"))
                preset_dropdown = gr.Dropdown(
                    label=i18n("Select Preset"),
                    choices=list_json_files(PRESETS_DIR),
                    interactive=True,
                )
                preset_name_input = gr.Textbox(label=i18n("Preset Name"))
                export_button = gr.Button(i18n("Save Preset"))
                import_file = gr.File(label=i18n("Import Preset"), type="filepath")
                refresh_presets_button = gr.Button(i18n("Refresh Presets"))
                clear_outputs = gr.Button(i18n("Clear Output Audios"))

        terms_checkbox = gr.Checkbox(
            label=i18n("I agree to the terms of use"),
            value=False,
            interactive=True,
        )
        convert_button = gr.Button(i18n("Convert"))
        vc_output1 = gr.Textbox(label=i18n("Output Information"))
        vc_output2 = gr.Audio(label=i18n("Export Audio"))

    # Batch inference tab
    with gr.Tab(i18n("Batch")):
        with gr.Column():
            input_folder_batch = gr.Textbox(
                label=i18n("Input Folder"),
                value=os.path.join(audio_root, "input"),
                interactive=True,
            )
            output_folder_batch = gr.Textbox(
                label=i18n("Output Folder"),
                value=audio_root,
                interactive=True,
            )
            export_format_batch = gr.Radio(
                label=i18n("Export Format"),
                choices=["WAV", "MP3", "FLAC"],
                value="WAV",
                interactive=True,
            )
            sid_batch = gr.Dropdown(
                label=i18n("Speaker ID"),
                choices=get_speakers_id(model_file.value),
                value=0,
                interactive=True,
            )
            with gr.Accordion(i18n("Settings"), open=False):
                pitch_batch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=i18n("Pitch"),
                    value=0,
                    interactive=True,
                )
                index_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Search Feature Ratio"),
                    value=0.75,
                    interactive=True,
                )
                rms_mix_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Volume Envelope"),
                    value=1,
                    interactive=True,
                )
                protect_batch = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protect Voiceless Consonants"),
                    value=0.5,
                    interactive=True,
                )
                split_audio_batch = gr.Checkbox(
                    label=i18n("Split Audio"),
                    value=False,
                    interactive=True,
                )
                autotune_batch = gr.Checkbox(
                    label=i18n("Autotune"),
                    value=False,
                    interactive=True,
                )
                autotune_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Autotune Strength"),
                    value=1,
                    interactive=True,
                    visible=False,
                )
                clean_audio_batch = gr.Checkbox(
                    label=i18n("Clean Audio"),
                    value=False,
                    interactive=True,
                )
                clean_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Clean Strength"),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                formant_shifting_batch = gr.Checkbox(
                    label=i18n("Formant Shifting"),
                    value=False,
                    interactive=True,
                )
                formant_qfrency_batch = gr.Slider(
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    label=i18n("Quefrency"),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                formant_timbre_batch = gr.Slider(
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    label=i18n("Timbre"),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                f0_method_batch = gr.Radio(
                    label=i18n("Pitch Extraction Algorithm"),
                    choices=["crepe", "rmvpe", "fcpe"],
                    value="rmvpe",
                    interactive=True,
                )
                hop_length_batch = gr.Slider(
                    minimum=1,
                    maximum=512,
                    step=1,
                    label=i18n("Hop Length"),
                    value=128,
                    interactive=True,
                    visible=False,
                )
                embedder_model_batch = gr.Radio(
                    label=i18n("Embedder Model"),
                    choices=["contentvec", "chinese-hubert-base", "japanese-hubert-base", "custom"],
                    value="contentvec",
                    interactive=True,
                )
                with gr.Column(visible=False) as embedder_custom_batch:
                    embedder_model_custom_batch = gr.Dropdown(
                        label=i18n("Custom Embedder"),
                        choices=refresh_embedders_folders(),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    refresh_embedders_button_batch = gr.Button(i18n("Refresh Embedders"))
                    folder_name_input_batch = gr.Textbox(label=i18n("Folder Name"))
                    bin_file_upload_batch = gr.File(label=i18n("Upload .bin"), type="filepath")
                    config_file_upload_batch = gr.File(label=i18n("Upload .json"), type="filepath")
                    move_files_button_batch = gr.Button(i18n("Move Files"))
                preset_dropdown_batch = gr.Dropdown(
                    label=i18n("Select Preset"),
                    choices=list_json_files(PRESETS_DIR),
                    interactive=True,
                )
                preset_name_input_batch = gr.Textbox(label=i18n("Preset Name"), placeholder=i18n("Enter preset name"))
                export_button_batch = gr.Button(i18n("Save Preset"))
                import_file_batch = gr.File(label=i18n("Import Preset"), type="filepath")
                refresh_presets_button_batch = gr.Button(i18n("Refresh Presets"))
                clear_outputs_batch = gr.Button(i18n("Clear Output Audios"))

        terms_checkbox_batch = gr.Checkbox(
            label=i18n("I agree to the terms of use"),
            value=False,
            interactive=True,
        )
        convert_button_batch = gr.Button(i18n("Convert"))
        stop_button = gr.Button(i18n("Stop"), visible=False)
        vc_output3 = gr.Textbox(label=i18n("Output Information"))

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_formant_shifting(checkbox):
        return (
            gr.update(visible=checkbox),
            gr.update(visible=checkbox),
        )

    def toggle_visible_hop_length(f0_method):
        return {"visible": f0_method in ["crepe", "crepe-tiny"], "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        return {"visible": embedder_model == "custom", "__type__": "update"}

    def enable_stop_convert_button():
        return {"visible": False, "__type__": "update"}, {"visible": True, "__type__": "update"}

    def disable_stop_convert_button():
        return {"visible": True, "__type__": "update"}, {"visible": False, "__type__": "update"}

    def enforce_terms(terms_accepted, *args):
        if not terms_accepted:
            gr.Info("You must agree to the Terms of Use to proceed.")
            return "Terms not accepted.", None
        return run_infer_script(*args)

    def enforce_terms_batch(terms_accepted, *args):
        if not terms_accepted:
            gr.Info("You must agree to the Terms of Use to proceed.")
            return "Terms not accepted.", None
        return run_batch_infer_script(*args)

    autotune.change(fn=toggle_visible, inputs=[autotune], outputs=[autotune_strength])
    clean_audio.change(fn=toggle_visible, inputs=[clean_audio], outputs=[clean_strength])
    formant_shifting.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[formant_qfrency, formant_timbre],
    )
    formant_shifting_batch.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting_batch],
        outputs=[formant_qfrency_batch, formant_timbre_batch],
    )
    f0_method.change(fn=toggle_visible_hop_length, inputs=[f0_method], outputs=[hop_length])
    f0_method_batch.change(fn=toggle_visible_hop_length, inputs=[f0_method_batch], outputs=[hop_length_batch])
    embedder_model.change(fn=toggle_visible_embedder_custom, inputs=[embedder_model], outputs=[embedder_custom])
    embedder_model_batch.change(fn=toggle_visible_embedder_custom, inputs=[embedder_model_batch], outputs=[embedder_custom_batch])
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, audio, sid, sid_batch],
    )
    audio.change(fn=output_path_fn, inputs=[audio], outputs=[output_path])
    upload_audio.upload(fn=save_to_wav2, inputs=[upload_audio], outputs=[audio, output_path])
    upload_audio.stop_recording(fn=save_to_wav, inputs=[upload_audio], outputs=[audio, output_path])
    clear_outputs.click(fn=delete_outputs, inputs=[], outputs=[])
    clear_outputs_batch.click(fn=delete_outputs, inputs=[], outputs=[])
    preset_dropdown.change(
        fn=update_sliders,
        inputs=[preset_dropdown],
        outputs=[pitch, index_rate, rms_mix_rate, protect],
    )
    preset_dropdown_batch.change(
        fn=update_sliders,
        inputs=[preset_dropdown_batch],
        outputs=[pitch_batch, index_rate_batch, rms_mix_rate_batch, protect_batch],
    )
    export_button.click(
        fn=export_presets_button,
        inputs=[preset_name_input, pitch, index_rate, rms_mix_rate, protect],
        outputs=[vc_output1],
    )
    export_button_batch.click(
        fn=export_presets_button,
        inputs=[preset_name_input_batch, pitch_batch, index_rate_batch, rms_mix_rate_batch, protect_batch],
        outputs=[vc_output3],
    )
    import_file.change(fn=import_presets_button, inputs=[import_file], outputs=[preset_dropdown])
    import_file_batch.change(fn=import_presets_button, inputs=[import_file_batch], outputs=[preset_dropdown_batch])
    refresh_presets_button.click(fn=refresh_presets, inputs=[], outputs=[preset_dropdown])
    refresh_presets_button_batch.click(fn=refresh_presets, inputs=[], outputs=[preset_dropdown_batch])
    move_files_button.click(
        fn=create_folder_and_move_files,
        inputs=[folder_name_input, bin_file_upload, config_file_upload],
        outputs=[vc_output1],
    )
    move_files_button_batch.click(
        fn=create_folder_and_move_files,
        inputs=[folder_name_input_batch, bin_file_upload_batch, config_file_upload_batch],
        outputs=[vc_output3],
    )
    refresh_embedders_button.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom],
    )
    refresh_embedders_button_batch.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom_batch],
    )
    convert_button.click(
        fn=enforce_terms,
        inputs=[
            terms_checkbox,
            pitch,
            index_rate,
            rms_mix_rate,
            protect,
            hop_length,
            f0_method,
            audio,
            output_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            autotune_strength,
            clean_audio,
            clean_strength,
            export_format,
            embedder_model,
            embedder_model_custom,
            formant_shifting,
            formant_qfrency,
            formant_timbre,
            sid,
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button_batch.click(
        fn=enforce_terms_batch,
        inputs=[
            terms_checkbox_batch,
            pitch_batch,
            index_rate_batch,
            rms_mix_rate_batch,
            protect_batch,
            hop_length_batch,
            f0_method_batch,
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            split_audio_batch,
            autotune_batch,
            autotune_strength_batch,
            clean_audio_batch,
            clean_strength_batch,
            export_format_batch,
            embedder_model_batch,
            embedder_model_custom_batch,
            formant_shifting_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
            sid_batch,
        ],
        outputs=[vc_output3],
    )
    convert_button_batch.click(
        fn=enable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
    stop_button.click(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )

    return inference_tab
