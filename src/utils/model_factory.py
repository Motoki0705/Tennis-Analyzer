from hydra.utils import get_class

from .load_model import (  # Assuming this is for PyTorch Lightning checkpoints
    load_model_weights,
)


def create_model_with_hf_load(
    model_class_name: str,
    pretrained_model_name_or_path: str,
    ckpt_path: str = None,
    **model_args,
):
    """
    Instantiates a Hugging Face model using from_pretrained and optionally loads custom weights.
    """
    model_class = get_class(
        model_class_name
    )  # e.g., transformers.ConditionalDetrForObjectDetection

    # Most HF models are loaded with from_pretrained
    if hasattr(model_class, "from_pretrained"):
        model = model_class.from_pretrained(pretrained_model_name_or_path, **model_args)
    else:  # Fallback for generic PyTorch models
        model = model_class(**model_args)

    if ckpt_path:
        # This assumes load_model_weights is for PyTorch/Lightning checkpoints
        # HF fine-tuned models might already have weights loaded by from_pretrained if ckpt_path points to HF hub or local HF save
        # Or, if ckpt_path is a PL checkpoint, this will adapt it.
        model = load_model_weights(model, ckpt_path)
    return model
