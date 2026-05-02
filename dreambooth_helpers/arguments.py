import argparse, random
from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1


def parse_arguments() -> JoePennaDreamboothConfigSchemaV1:
    def _get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config_file_path",
            type=str,
            required=False,
            default=None,
            help="A config file containing all of your variables"
        )
        parser.add_argument(
            "--project_name",
            type=str,
            required=False,
            help="Name of the project"
        )
        parser.add_argument(
            "--debug",
            action='store_true',
            help="Enable debug logging",
        )
        parser.add_argument(
            "--token",
            type=str,
            required=False,
            help="Unique token you want to represent your trained model. will resolve token name from directory structure automatically "
        )
        parser.add_argument(
            "--token_only",
            action='store_true',
            help="Train only using the token and no class."
        )
        parser.add_argument(
            "--training_model",
            type=str,
            required=False,
            default='https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt',
            help="Path to a local model or the url of a huggingface model repo to use for training (e.g 'v1-5-pruned.ckpt' -- 'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt')"
        )
        parser.add_argument(
            "--training_images",
            type=str,
            required=True,
            help="Path to training images directory"
        )
        parser.add_argument(
            "--regularization_images",
            type=str,
            required=False,
            help="Path to directory with regularization images"
        )
        parser.add_argument(
            "--class_word",
            type=str,
            required=False,
            help="Match class_word to the category of images you want to train. Example: 'man', 'woman', 'dog', or 'artstyle'."
        )
        parser.add_argument(
            "--flip_p",
            type=float,
            required=False,
            default=0.5,
            help="Flip Percentage "
                 "Example: if set to 0.5, will flip (mirror) your training images 50% of the time."
                 "This helps expand your dataset without needing to include more training images."
                 "This can lead to worse results for face training since most people's faces are not perfectly symmetrical."
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            required=False,
            default=1.0e-06,
            help="Set the learning rate. Defaults to 1.0e-06 (0.000001).  Accepts scientific notation."
        )
        parser.add_argument(
            "--save_every_x_steps",
            type=int,
            required=False,
            default=0,
            help="Saves a checkpoint every x steps"
        )
        parser.add_argument(
            "--gpu",
            type=int,
            default=0,
            required=False,
            help="Specify a GPU other than 0 to use for training.  Multi-GPU support is not currently implemented."
        )
        parser.add_argument(
            "--repeats",
            type=int,
            default=100,
            required=False
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            required=False
        )
        parser.add_argument(
            "--accumed_grads",
            type=int,
            default=1,
            required=False
        )
        parser.add_argument(
            "--resolution",
            type=int,
            default=512,
            required=False
        )
        parser.add_argument(
            '--center_crop',
            action='store_true'
        )
        parser.add_argument(
            '--fp32',
            type=str,
            nargs='?',
            const='float32',
            default='float16',
            help='saves model state dict as float32, rather than float16 (the default)'
        )
        parser.add_argument(
            '--safetensors',
            action='store_true'
        )

        return parser

    parser = _get_parser()
    opt, unknown = parser.parse_known_args()
    config = JoePennaDreamboothConfigSchemaV1()

    if opt.config_file_path is not None:
        config.saturate_from_file(config_file_path=opt.config_file_path)
    else:
        config.saturate(
            project_name=opt.project_name,
            debug=opt.debug,
            gpu=opt.gpu,
            save_every_x_steps=opt.save_every_x_steps,
            training_images_folder_path=opt.training_images,
            regularization_images_folder_path=opt.regularization_images,
            token=opt.token,
            token_only=opt.token_only,
            class_word=opt.class_word,
            flip_percent=opt.flip_p,
            learning_rate=opt.learning_rate,
            model_path=opt.training_model,
            repeats=opt.repeats,
            batch_size=opt.batch_size,
            accumed_grads=opt.accumed_grads,
            res=opt.resolution,
            crop=opt.center_crop,
            precision=opt.fp32,
            safetensors=opt.safetensors
        )

    return config
