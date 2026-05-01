import os, json, math, glob, shutil, sys, torch
from datetime import datetime, timezone
from pytorch_lightning import seed_everything


class JoePennaDreamboothConfigSchemaV1():
    def __init__(self):
        self.schema = 1
        self.config_date_time = datetime.now(timezone.utc).strftime("%m-%d-%Y")
        self.project_name = None
        self.token = None
        self.token_only = False
        self.class_word = None
        self.training_images_folder_path = None
        self.regularization_images_folder_path = None
        self.model_path = None
        self.precision = None
        self.safetensors = False
        self.repeats = 100
        self.learning_rate = 1.0e-06
        self.batch_size = 1
        self.accumed_grads = 1
        self.res = 512
        self.crop = False
        self.flip_percent = 0.5
        self.seed = 1337
        self.save_every_x_steps = 0
        self.gpu = 0
        self.debug = False        
        self.model_repo_id = None

    def saturate(
        self,
        project_name,
        token,
        token_only,
        class_word,
        training_images_folder_path,
        regularization_images_folder_path,
        model_path,
        precision,
        safetensors,
        repeats,
        learning_rate,
        batch_size,
        accumed_grads,
        res,
        crop,
        flip_percent,
        save_every_x_steps,
        seed,
        gpu,
        debug
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
        
        seed_everything(self.seed)

        if self.save_every_x_steps < 0:
            raise Exception("--save_every_x_steps: must be greater than or equal to 0")

        self.training_images_folder_path = os.path.relpath(training_images_folder_path)

        if not os.path.exists(self.training_images_folder_path):
            raise Exception(f"Training Images Path Not Found: '{self.training_images_folder_path}'.")

        tkns = os.listdir(self.training_images_folder_path)
        tkn_cls = {0: {'token': tkns[0]}, 1: {'token': tkns[1]}}
        for n in [0, 1]:
            tkn_cls[n]['class'] = os.listdir(os.path.join(self.training_images_folder_path, tkn_cls[n]['token']))

        self.training_images = [os.path.relpath(f, sys.path[0]) for f in
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpg'), recursive=True) +
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpeg'), recursive=True) +
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.png'), recursive=True)
        ]

        self.training_images_count = len(self.training_images)
        if self.training_images_count <= 0:
            raise Exception(f"No Training Images (*.png, *.jpg, *.jpeg) found in '{self.training_images_folder_path}'.")
        self.max_training_steps = int(self.training_images_count * self.repeats / (self.batch_size * self.accumed_grads))

        if not self.token_only:
            self.regularization_images_folder_path = os.path.relpath(regularization_images_folder_path)
            self.class_word = tkn_cls[0]['class']

        if not os.path.exists(self.regularization_images_folder_path):
            raise Exception(f"Regularization Images Path Not Found: '{self.regularization_images_folder_path}'.")

        self.token = tkn_cls[0]['token']
        if self.token is None or self.token == '':
            raise Exception(f"Token not provided.")

        self.flip_percent = flip_percent
        if self.flip_percent < 0 or self.flip_percent > 1:
            raise Exception("--flip_p: must be between 0 and 1")

        self.learning_rate = learning_rate
        self.precision = precision
        if safetensors:
            self.model_format = '.safetensors'
        else: self.model_format = '.ckpt'

        self.project_name = f"{tkn_cls[0]['token']}-{tkn_cls[0]['class']}_{tkn_cls[1]['token']}-{tkn_cls[1]['class']}"
        self.project_config_filename = f"{self.project_name}-config.json"
        
        if not os.path.exists(os.path.relpath(model_path)):
            if model_path.startswith('https://huggingface.co') and not os.path.exists(os.path.basename(model_path)):
                self.model_path = self.get_model_from_hub(repo_url=model_path)    
            else: raise Exception(f"Model Path Not Found: '{model_path}'.")
        else: self.model_path = os.path.relpath(model_path)
            
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
        title = f"{date_string}_{self.project_name}_{int(steps):05d}_steps".replace(' ', '_') + self.model_format
        return title

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

    def get_model_from_hub(self, repo_url=None):
        from huggingface_hub.file_download import hf_hub_download
        import hf_xet
        repo_id = f"{repo_url.split('/')[3]}/{repo_url.split('/')[4]}"
        model_ckpt = os.path.basename(repo_url)
        print(f"Downloading '{model_ckpt}'")
        hf_hub_download(repo_id, model_ckpt, local_dir=sys.path[0])
        return os.path.join(sys.path[0], model_ckpt)
        
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

