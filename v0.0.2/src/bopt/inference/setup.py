import os.path

from bopt.data import preprocessors
from bopt.inference import ClassificationInferenceSetup
from bopt.modeling import load_model
from bopt.modeling.classifier import Classifier
from bopt.unigram_lm_tokenizers.loading import load_input_tokenizer, load_label_tokenizer
from bopt.utils import load_vocab
from experiments.utils.datasets import list_collate

from torch.utils.data import DataLoader, SequentialSampler

def setup_classification(args):

    if os.path.exists(args.output_directory) and not args.overwrite_output_directory:
        raise ValueError("Please set overwrite_output_directory to true when using existing directories.")

    # load vocabularies
    input_vocab = load_vocab(args.input_vocab)
    input_vocab.specials = set(args.special_tokens)
    output_vocab = load_vocab(args.output_vocab)

    # load datasets
    dataset = preprocessors[args.domain](args.dataset, args)


    dataloader = DataLoader(dataset,
                                  sampler=SequentialSampler(dataset),
                                  batch_size=args.gpu_batch_size,
                                  num_workers=args.data_num_workers,
                                  collate_fn=list_collate)

    # model
    model, config = load_model(config=os.path.join(args.model_path, "config.json"), pad_token_id=None, bias_mode=args.bias_mode, saved_model=args.model_path, ignore=set())

    # tokenizer
    input_tokenizer = load_input_tokenizer(args.input_tokenizer_model, args.input_tokenizer_mode, input_vocab, log_space_parametrization=args.log_space_parametrization, weight_file=args.input_tokenizer_weights,
                                           num_hidden_layers=args.nulm_num_hidden_layers, hidden_size=args.nulm_hidden_size, model=model)
    label_tokenizer = load_label_tokenizer(args.input_tokenizer_mode, output_vocab)

    # classifier
    classifier = Classifier(model, input_tokenizer, label_tokenizer).to(args.device)

    specials = set(args.special_tokens)

    return ClassificationInferenceSetup(args=args,
                                       dataloader=dataloader,
                                       classifier=classifier,
                                       specials=specials)