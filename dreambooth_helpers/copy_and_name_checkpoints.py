import os
import re
import shutil
import glob
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1

def copy_and_name_checkpoints(
    config: JoePennaDreamboothConfigSchemaV1,
):
    output_folder = config.trained_models_directory()
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # copy the config file to the directory as well
    config.save_config_to_file(
        save_path=output_folder
    )

    logs_directory = config.log_directory()
    if not os.path.exists(logs_directory):
        print(f"No checkpoints found in {logs_directory}")
        return

    # Gather, name, and move the checkpoint files.
    checkpoints_and_steps = []
    if config.save_every_x_steps == 0:
        checkpoints_and_steps.append(
            (
                os.path.join(config.log_checkpoint_directory(), f"last{config.format}"),
                str(config.max_training_steps)
            )
        )
    else:
        intermediate_checkpoints_directory = config.log_intermediate_checkpoints_directory()
        file_paths = glob.glob(os.path.join(intermediate_checkpoints_directory, f"*{config.format}"))

        for i, original_file_path in enumerate(file_paths):
            # Grab the steps from the filename
            # "epoch=000000-step=000000250.ckpt" => "250.ckpt"
            # 'logs\\2023-03-11T20-03-37_ap_v15vae_ultrahq\\ckpts\\trainstep_ckpts\\epoch=000000-step=000000250.ckpt'
            file_name = os.path.basename(original_file_path)
            checkpoint_steps = re.sub(r"epoch=\d{6}-step=0*", "", file_name)

            # Remove the .ckpt
            # "250.ckpt" => "250"
            checkpoint_steps = checkpoint_steps.replace(f"{config.format}", "")
            checkpoints_and_steps.append(
                (
                    original_file_path,
                    checkpoint_steps
                )
            )
    checkpoints_found = True
    for i, file_and_steps in enumerate(checkpoints_and_steps):

        original_file_name, steps = file_and_steps[0], file_and_steps[1]

        # Setup the filenames
        new_file_name = config.create_checkpoint_file_name(steps)
        output_file_name = os.path.join(output_folder, new_file_name)
    
        if os.path.exists(original_file_name):
            print(f"Moving {original_file_name} to {output_file_name}")
            shutil.move(original_file_name, output_file_name)
        else: checkpoints_found = False
        

    if checkpoints_found:
        print(f"✅ Download your trained model(s) from the '{output_folder}' folder and use in your favorite Stable Diffusion repo!")
    else:
        print("No checkpoints found.")

