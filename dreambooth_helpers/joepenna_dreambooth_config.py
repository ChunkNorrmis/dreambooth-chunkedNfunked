import os, json, math, glob, shutil, sys, torch, random, safetensors.torch, argparse
from datetime import datetime, timezone
from pytorch_lightning import seed_everything


class JoePennaDreamboothConfigSchemaV1():
    def __init__(self, opts):
        opt, ukopt = opts()
        
        self.schema = 1
        self.config_date_time = datetime.now(timezone.utc).strftime("%m-%d-%Y")
        self.token = opt.token
        self.token_only = opt.token_only
        self.class_word = opt.class_word
        self.training_images_folder_path = opt.training_images
        self.regularization_images_folder_path = opt.regularization_images
        self.precision = opt.fp32
        self.repeats = opt.repeats
        self.learning_rate = opt.learning_rate
        self.batch_size = opt.batch_size
        self.accumed_grads = opt.accumed_grads
        self.res = opt.resolution
        self.crop = opt.center_crop
        self.flip_percent = opt.flip_p
        self.seed = random.randrange(1, 1e+05)
        self.save_every_x_steps = opt.save_every_x_steps
        self.gpu = opt.gpu
        self.debug = opt.debug

        seed_everything(self.seed)
        
        tkns = os.listdir(self.training_images_folder_path)
        tkn_cls = {0: {'token': tkns[0]}, 1: {'token': tkns[1]}}
        for n in [0, 1]:
            tkn_cls[n]['class'] = os.listdir(os.path.join(self.training_images_folder_path, tkn_cls[n]['token']))
        
        self.training_images = [os.path.relpath(f) for f in
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpg'), recursive=True) +
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpeg'), recursive=True) +
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.png'), recursive=True)
        ]

        self.training_images_count = len(self.training_images)
        if self.training_images_count <= 0:
            raise Exception(f"No Training Images (*.png, *.jpg, *.jpeg) found in '{self.training_images_folder_path}'.")
        self.max_training_steps = int(self.training_images_count * self.repeats / (self.batch_size * self.accumed_grads))

        if opt.savetensors:
            self.model_format = '.safetensors'
        else: self.model_format = '.ckpt'

        self.project_name = f"{self.token}-{self.class_word}_{tkn_cls[0]['token']}-{tkn_cls[0]['class']}"
        self.project_config_filename = f"{self.project_name}-config.json"

        if not os.path.exists(opt.model_path):
            if opt.model_path.startswith('https://huggingface.co'):
                self.model_path = self.get_model_from_hub(repo_url=opt.model_path)    
            else: raise Exception(f"Model Path Not Found: '{opt.model_path}'.")
        else: self.model_path = opt.model_path

        self.opts = opt
        self.validate_gpu_vram()
        self._create_log_folders()
    
    def __call__(self):
        return self.opts
        

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
        default_path = os.path.join(sys.path[0], model_ckpt)
        if not os.path.exists(default_path):
            print(f"Downloading '{model_ckpt}'")
            hf_hub_download(repo_id, model_ckpt, local_dir=sys.path[0])
        return default_path
        
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

