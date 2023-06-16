from bopt.bilevel import ClassificationBilevelTrainingSetup
from bopt.modeling.classifier import ClassifierOutput



def train_loss_inner(setup: ClassificationBilevelTrainingSetup, backward=False):
    loss = 0
    batch_counter = 0
    for i, batch in enumerate(setup.train_inner_dataloader):
        if i > setup.args.train_batch_size_inner // setup.args.gpu_batch_size_inner: break # only do one training batch max
        # training
        ids, sentences, labels = batch

        # run model
        output: ClassifierOutput = setup.classifier(setup, ids, sentences, labels, "train_inner")

        # compute loss
        loss += output.task_loss

        batch_counter += 1

    return loss / batch_counter


