import os, re, shutil, glob, torch
from tqdm.auto import tqdm
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1
from ldm.depicklizer import depicklize


def copy_and_name_checkpoints(config: JoePennaDreamboothConfigSchemaV1):
    checkpoints_found = False
    output_folder = config.trained_models_directory()
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    logs_directory = config.log_directory()
    log_ckpt_dir = config.log_checkpoint_directory()
    intermediate_checkpoints_directory = config.log_intermediate_checkpoints_directory()
    file_paths = [os.path.relpath(cp) for cp in glob.glob(os.path.join(intermediate_checkpoints_directory, '*.ckpt')) + [os.path.join(log_ckpt_dir, 'last.ckpt')]]
    config.save_config_to_file(save_path=output_folder)
    if not os.path.exists(logs_directory):
        print(f"No checkpoints found in {logs_directory}")
        return
    
    if config.model_format == '.safetensors':
        print(f"Depickling model checkpoint(s)")
        print(' ')
    p_bar = tqdm(total=len(file_paths))
    for file_path in file_paths:
        checkpoints_found = True
        if os.path.basename(file_path) == 'last.ckpt':
            if int(torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)['global_step']) == config.max_training_steps:
                last = os.path.join(output_folder, config.create_checkpoint_file_name(config.max_training_steps))
                if config.model_format == '.safetensors':
                    depicklize(file_path, nil_pickle=last)
                else:
                    shutil.move(file_path, last)
                p_bar.update()
        else:
            file_name = os.path.basename(file_path)
            steps = re.sub(r"epoch=\d{6}-step=0*", "", file_name)
            steps = os.path.splitext(steps)[0]
            output_file = os.path.join(output_folder, config.create_checkpoint_file_name(steps))
            if config.model_format == '.safetensors':
                depicklize(file_path, nil_pickle=output_file)
            else:
                shutil.move(file_path, output_file)
            p_bar.update()
            
    if checkpoints_found:
        print(f"✅ Model(s) moved to '{output_folder}'")
    else:
        print("No checkpoints found.")


