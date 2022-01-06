import sys
sys.path.append('..')
sys.path.append('.')

from ImageClassification.model.CoAtNet import CoAtNet
import ImageClassification.utils.ImageLoading as ImageLoading
from ImageClassification.model.CoAtNet import weight_init
from ImageClassification.model.Inceptionv3 import inception

import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

'''
Something for training
'''


def train_loop(dataloader, model, loss_fn, optimizer, scheduler, args):
    size = len(dataloader.dataset)


    for batch, (_, X, y) in enumerate(dataloader):
        if args.GPU:
            device = torch.device('cuda')
            X = X.to(device)
            y = y.to(device)

        pred = model(X).logits

        # pred = model(X)
        '''
        plt.subplot(1, 2, 1)
        plt.imshow(np.moveaxis(X.cpu().detach().numpy()[0], 0, -1))
        plt.subplot(1, 2, 2)
        plt.imshow(np.moveaxis(X.cpu().detach().numpy()[1], 0, -1))
        plt.show()
        '''
        # print(pred.cpu().detach().numpy())
        print(pred.argmax(1).cpu().numpy(), y.argmax(1).cpu().numpy())

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    scheduler.step()

def tes_loop(dataloader, model, loss_fn, args):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for _, X, y in dataloader:
            if args.GPU:
                device = torch.device('cuda')
                X = X.to(device)
                y = y.to(device)

            pred = model(X)
            print(pred.argmax(1).cpu().numpy(), y.argmax(1).cpu().numpy())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.5f}%, Avg loss: {test_loss:>8f} \n")

    return correct, test_loss


def train(args):
    print('\n------------------Now training------------------')
    learning_rate = args.lr
    EPOCHS = args.epochs
    image_size = args.img_size
    classes = args.classes
    dropout_rate = args.dropout_rate
    weight_decay = args.weight_decay

    # coatnet = CoAtNet(in_ch=3, image_size=image_size, dropout_rate=dropout_rate, classes=classes)
    # coatnet.apply(weight_init)
    model = inception(args)

    if args.GPU:
        device = torch.device('cuda')
        # coatnet.to(device)
        model.to(device)

    PASS = ImageLoading.PassTheData(args)
    train_dataloader = PASS.pass_train_dataloader()
    test_dataloader = PASS.pass_test_dataloader()

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(coatnet.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD(coatnet.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.5)
    # optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    history = {'accuracy':[], 'loss':[]}
    for t in range(EPOCHS):
        print('\n')
        print(f"Epoch {t+1}\n-------------------------------")
        # for f in => K-fold
        # coatnet.train()
        # train_loop(train_dataloader, coatnet, criterion, optimizer, scheduler, args)
        # coatnet.eval()
        # correct, test_loss = tes_loop(test_dataloader, coatnet, criterion, args)
        model.train()
        train_loop(train_dataloader, model, criterion, optimizer, scheduler, args)
        model.eval()
        correct, test_loss = tes_loop(test_dataloader, model, criterion, args)

        history['accuracy'].append(correct)
        history['loss'].append(test_loss)
        if t == 0:
            # torch.save(coatnet, './coatnet_net.pt')
            torch.save(model, './inception_net.pt')
        else:
            if correct > max(history['accuracy']):
                # torch.save(coatnet, './coatnet_net.pt')
                torch.save(model, './inception_net.pt')



    plt.subplot(1, 2, 1)
    plt.title('Accuracy')
    plt.plot(range(EPOCHS), history['accuracy'])
    plt.subplot(1, 2, 2)
    plt.title('loss')
    plt.plot(range(EPOCHS), history['loss'])
    plt.show()
    print("Done!")