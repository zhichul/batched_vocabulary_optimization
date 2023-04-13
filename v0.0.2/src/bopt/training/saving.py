import json
import os

import torch

from bopt.training import TrainingState


def save_classification_checkpoint(output_dir, checkpoint_name, state: TrainingState, classifier):
    checkpointdir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpointdir, exist_ok=True)

    # save tokenizer
    classifier.input_tokenizer.save_to_folder(checkpointdir)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(checkpointdir, "pytorch_model.bin")
    output_config_file = os.path.join(checkpointdir, "config.json")

    # save model and config
    torch.save(classifier.model.state_dict(), output_model_file)
    classifier.model.config.to_json_file(output_config_file)

    with open(os.path.join(checkpointdir, "info.json"), "wt") as f:
        print(json.dumps({"step": state.step, "epoch": state.epoch}), file=f)
