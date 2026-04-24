import json
import math
import os
import glob
import shutil
import sys
import torch
from datetime import datetime, timezone
from pytorch_lightning import seed_everything


class JoePennaDreamboothConfigSchemaV1():
    def __init__(self, schema=1):
        self.schema = schema

    def saturate(
            self,
            project_name=None,
            save_every_x_steps=0,
            training_images_folder_path=None,
            regularization_images_folder_path=None,
            token=None,
            class_word=None,
            flip_percent=0.5,
            learning_rate=1e-06,
            model_path=None,
            repeats=100,
            batch_size=1,
            accumed_grads=1,
            res=512,
            crop=True,
            seed=1337,
            token_only=False,
            debug=False,
            gpu=0,
            precision='float16',
            safetensors=False,
            model_repo_id=None,
            run_seed_everything=True,
            config_date_time=None,
    ):

        self.repeats = repeats
        self.batch_size = batch_size
        self.accumed_grads = accumed_grads
        self.res = res
        self.crop = crop
        self.seed = seed
        self.save_every_x_steps = save_every_x_steps
        self.debug = debug
        self.gpu = gpu
        self.token_only = token_only

        if config_date_time is None:
            self.config_date_time = datetime.now(timezone.utc).strftime("%m-%d-%Y")
        else:
            self.config_date_time = config_date_time
            
        if run_seed_everything:
            seed_everything(self.seed)

        if self.save_every_x_steps < 0:
            raise Exception("--save_every_x_steps: must be greater than or equal to 0")

        self.training_images_folder_path = os.path.relpath(training_images_folder_path)

        if not os.path.exists(self.training_images_folder_path):
            raise Exception(f"Training Images Path Not Found: '{self.training_images_folder_path}'.")

        self.tokens = [tk.split('/')[-2] for tk in glob.glob(f"{self.training_images_folder_path}/**/*")]
        self.classes = [cl.split('/')[-1] for cl in glob.glob(f"{self.training_images_folder_path}/**/*")]
        
        self.training_images = [os.path.relpath(f, sys.path[0]) for f in
                                 glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpg'), recursive=True) +
                                 glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpeg'), recursive=True) +
                                 glob.glob(os.path.join(self.training_images_folder_path, '**', '*.png'), recursive=True)
                             ]
        #_training_image_paths = [os.path.relpath(f, self.training_images_folder_path) for f in _training_image_paths]

        self.training_images_count = len(self.training_images)
        if self.training_images_count <= 0:
            raise Exception(f"No Training Images (*.png, *.jpg, *.jpeg) found in '{self.training_images_folder_path}'.")
        self.max_training_steps = int(self.training_images_count * self.repeats / (self.batch_size * self.accumed_grads))
        
        #self.training_images = _training_image_paths
        if not self.token_only and regularization_images_folder_path is not None and regularization_images_folder_path != '':
            self.regularization_images_folder_path = os.path.relpath(regularization_images_folder_path)
            #self.r_token = os.path.basename((self.regularization_images_folder_path))
            #self.r_class_word = os.path.basename(os.listdir(os.path.join(self.regularization_images_folder_path, self.r_token)))

        if not os.path.exists(self.regularization_images_folder_path):
            raise Exception(f"Regularization Images Path Not Found: '{self.regularization_images_folder_path}'.")

        self.token = self.tokens[1]
        if self.token is None or self.token == '':
            raise Exception(f"Token not provided.")

        if not self.token_only:
            self.class_word = self.classes[1]
        else: self.token = self.classes[0]

        self.flip_percent = flip_percent
        if self.flip_percent < 0 or self.flip_percent > 1:
            raise Exception("--flip_p: must be between 0 and 1")

        self.learning_rate = learning_rate
        self.model_repo_id = model_repo_id
        self.precision = precision
        if safetensors:
            self.format = 'safetensors'
        else: self.format = 'ckpt'
        
        self.project_name = project_name
        self.project_config_filename = f"{self.config_date_time}-{self.project_name}-config.json"
        
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise Exception(f"Model Path Not Found: '{self.model_path}'.")

        self.validate_gpu_vram()
        self._create_log_folders()
        

    
    def validate_gpu_vram(self):
        def convert_size(size_bytes):
            if size_bytes == 0:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return "%s %s" % (s, size_name[i])

            # Check total available GPU memory

        gpu_vram = int(torch.cuda.get_device_properties(self.gpu).total_memory)
        print(f"gpu_vram: {convert_size(gpu_vram)}")
        twenty_one_gigabytes = 22548578304
        if gpu_vram < twenty_one_gigabytes:
            raise Exception(f"VRAM: Currently unable to run on less than {convert_size(twenty_one_gigabytes)} of VRAM.")

    def saturate_from_file(
            self,
            config_file_path: str,
    ):
        if not os.path.exists(config_file_path):
            print(f"{config_file_path} not found.", file=sys.stderr)
            return None
        else:
            config_file = open(config_file_path)
            config_parsed = json.load(config_file)

            if config_parsed['schema'] == 1:
                self.saturate(
                    project_name=config_parsed['project_name'],
                    max_training_steps=config_parsed['max_training_steps'],
                    save_every_x_steps=config_parsed['save_every_x_steps'],
                    training_images_folder_path=config_parsed['training_images_folder_path'],
                    regularization_images_folder_path=config_parsed['regularization_images_folder_path'],
                    token=config_parsed['token'],
                    class_word=config_parsed['class_word'],
                    flip_percent=config_parsed['flip_percent'],
                    learning_rate=config_parsed['learning_rate'],
                    model_path=config_parsed['model_path'],
                    config_date_time=config_parsed['config_date_time'],
                    seed=config_parsed['seed'],
                    debug=config_parsed['debug'],
                    gpu=config_parsed['gpu'],
                    model_repo_id=config_parsed['model_repo_id'],
                    token_only=config_parsed['token_only'],
                )
            else:
                print(f"Unrecognized schema: {config_parsed['schema']}", file=sys.stderr)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def create_checkpoint_file_name(self, steps: str):
        date_string = datetime.now(timezone.utc).strftime('%m-%d-%Y')
        return f"{date_string}_{self.project_name}_{int(steps):05d}_steps.{self.format}".replace(' ', '_')

    def save_config_to_file(
            self,
            save_path: str,
            create_active_config: bool = False,
    ):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        project_config_json = self.toJSON()
        config_save_path = os.path.join(save_path, self.project_config_filename)
        with open(config_save_path, "w") as config_file:
            config_file.write(project_config_json)

        if create_active_config:
            shutil.copy(config_save_path, os.path.join(save_path, "active-config.json"))
            print(project_config_json)
            print(f"✅ {self.project_config_filename} successfully generated.  Proceed to training.")

    def get_training_folder_name(self) -> str:
        return f"{self.config_date_time}_{self.project_name}"

    def log_directory(self) -> str:
        return os.path.join("logs", self.get_training_folder_name())

    def log_checkpoint_directory(self) -> str:
        return os.path.join(self.log_directory(), "ckpts")

    def log_intermediate_checkpoints_directory(self) -> str:
        return os.path.join(self.log_checkpoint_directory(), "trainstep_ckpts")

    def log_config_directory(self) -> str:
        return os.path.join(self.log_directory(), "configs")

    def trained_models_directory(self) -> str:
        return "trained_models"

    def _create_log_folders(self):
        os.makedirs(self.log_directory(), exist_ok=True)
        os.makedirs(self.log_checkpoint_directory(), exist_ok=True)
        os.makedirs(self.log_config_directory(), exist_ok=True)
        os.makedirs(self.trained_models_directory(), exist_ok=True)

