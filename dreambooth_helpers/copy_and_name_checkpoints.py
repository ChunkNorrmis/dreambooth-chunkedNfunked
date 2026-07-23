import os, re, shutil, glob, torch
from tqdm import tqdm
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1
from ldm.depicklizer import depicklize


def copy_and_name_checkpoints(config: JoePennaDreamboothConfigSchemaV1):
    output_folder = config.trained_models_directory()
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    logs_directory = config.log_directory()
    log_ckpt_dir = config.log_checkpoint_directory()
    intermediate_checkpoints_directory = config.log_intermediate_checkpoints_directory()
    model_paths = [os.path.relpath(mp) for mp in glob.glob(os.path.join(intermediate_checkpoints_directory, '*.ckpt')) + [os.path.join(log_ckpt_dir, 'last.ckpt')]]
    config.save_config_to_file(save_path=output_folder)
    if not os.path.exists(logs_directory):
        print(f"No checkpoints found in {logs_directory}")
        return
    
    if len(model_paths) > 0:
        if config.safetensors:
            print(f"Depickling model checkpoint(s)")
        for model_path in tqdm(model_paths):
            if os.path.basename(model_path) == 'last.ckpt':
                if int(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)['global_step']) == config.max_training_steps:
                    output_file = os.path.join(output_folder, config.create_checkpoint_file_name(config.max_training_steps))
            else:
                file_name = os.path.basename(model_path)
                steps = re.sub(r"epoch=\d{6}-step=0*", "", file_name).replace('.ckpt', '')
                #steps = os.path.splitext(steps)[0]
                output_file = os.path.join(output_folder, config.create_checkpoint_file_name(steps))
            if config.safetensors:
                depicklize(model_path, nil_pickle=output_file)
            else:
                shutil.move(model_path, output_file)
        print(f"✅ Model(s) moved to '{output_folder}'")
    else:
        print("No checkpoints found.")


