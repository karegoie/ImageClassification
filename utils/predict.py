import torch
import ImageClassification.utils.ImageLoading as ImageLoading
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def predict(args):

    print('\n------------------Now predicting------------------')
    # coatnet = torch.load('./coatnet_net.pt')
    # coatnet.eval()
    model = torch.load('./inception_net.pt')
    model.eval()

    if args.GPU:
        device = torch.device('cuda')
        # coatnet.to(device)
        model.to(device)

    PASS = ImageLoading.PassTheData(args)

    test_dataloader = PASS.pass_test_dataloader()
    predict_dataloader = PASS.pass_predict_dataloader()

    with torch.no_grad():
        names = []
        preds = []
        reals = [] # if real
        print('\n------------predicting---------------')
        right, total = 0, 0
        for name, Xs, ys in tqdm(predict_dataloader): # for name, Xs, ys in tqdm(test_dataloader):
            # print(name, Xs, ys)

            if args.GPU:
                device = torch.device('cuda')
                Xs = Xs.to(device)
                ys = ys.to(device)

            # pred = coatnet(Xs)
            pred = model(Xs)
            # print(pred.cpu().numpy())
            pred = pred.argmax(1).cpu().numpy()
            ys = ys.argmax(1).cpu().numpy()

            for n, p, y in zip(name, pred, ys):
                if n not in names:
                    names.append(n)
                    preds.append(p)
                    reals.append(y) # if real
                    if p==y: right+=1
                    total+=1

    print(f'\npredict accuracy is {right/total * 100}%')
    answers = {
        'hash' : names,
        'predict' : preds,
        'answer' : reals # if real
    }

    # print(len(answers['hash']), len(answers['predict']), len(answers['answer']))
    answers_df = pd.DataFrame(answers)
    answers_df.to_csv('./predict.csv', index=False)

    cf_matrix = confusion_matrix(answers['predict'], answers['answer'])
    classes = [f'{i}' for i in range(args.classes)]

    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix), index=classes, columns=classes)

    sns.heatmap(df_cm, annot=True)
    plt.xlabel('answer')
    plt.ylabel('prediction')
    plt.show()