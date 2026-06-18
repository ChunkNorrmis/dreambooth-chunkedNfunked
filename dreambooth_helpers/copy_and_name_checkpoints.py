import os, re, shutil, glob, torch
from tqdm import tqdm
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1
from ldm.depicklizer import depicklize


def copy_and_name_checkpoints(config: JoePennaDreamboothConfigSchemaV1):
    checkpoints_found = False
    output_folder = config.trained_models_directory()
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    config.save_config_to_file(save_path=output_folder)
    ckpt_dir = config.log_checkpoint_directory()
    intermediate_checkpoints_directory = config.log_intermediate_checkpoints_directory()
    first = os.path.join(ckpt_dir, 'last.ckpt')
    file_paths = glob.glob(os.path.join(intermediate_checkpoints_directory, '*.ckpt')) + [first]
    logs_directory = config.log_directory()
    if config.save_every_x_steps > 0:
        checkpoints_and_steps = []
    if not os.path.exists(logs_directory):
        print(f"No checkpoints found in {logs_directory}")
        return
    
    for i, original_file_path in tqdm(enumerate(file_paths), unit='conversion'):
        if original_file_path == first:
            checkpoints_found = True
            if int(torch.load(model, map_location=torch.device('cpu'), weights_only=False)['global_step']) == config.max_training_steps:
                last = os.path.join(output_folder, config.create_checkpoint_file_name(config.max_training_steps))
                if config.model_format == '.safetensors':
                    last = last.replace('.ckpt', '.safetensors')
                    depicklize(first, nil_pickle=last)
                else: shutil.move(first, last)
        else:
            file_name = os.path.basename(original_file_path)
            checkpoint_steps = re.sub(r"epoch=\d{6}-step=0*", "", file_name)
            checkpoint_steps = os.path.splitext(checkpoint_steps)[0]
            checkpoints_and_steps.append(original_file_path, checkpoint_steps)
            for i, file_and_steps in enumerate(checkpoints_and_steps):
                original_file_name, steps = file_and_steps[0], file_and_steps[1]
                checkpoints_found = True
                new_file_name = config.create_checkpoint_file_name(steps)
                output_file_name = os.path.join(output_folder, new_file_name)
                if config.model_format == '.safetensors':
                    output_file_name = output_file_name.replace('.ckpt', config.model_format)
                    depicklize(original_file_name, nil_pickle=output_file_name)
                else: shutil.move(original_file_name, output_file_name)
            
    if checkpoints_found:
        print(f"✅ Model(s) moved to './{output_folder}'")
    else:
        print("No checkpoints found.")
