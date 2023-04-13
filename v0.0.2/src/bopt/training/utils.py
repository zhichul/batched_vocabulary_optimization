def load_forever(dataloader):
    epoch = 0
    while True:
        for batch in dataloader:
            yield epoch, batch
        epoch += 1