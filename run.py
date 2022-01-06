import sys
sys.path.append('.')
sys.path.append('..')
import warnings
warnings.filterwarnings(action='ignore')

from ImageClassification.utils.train_part import train
import argparse
from ImageClassification.utils.predict import predict

def parse():
    parser = argparse.ArgumentParser(description='Dough ImageClassification project', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU', type=int, default=1, help='Whether GPU to allocate or not')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size, should be more than 1')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-c', '--classes', type=int, default=8, help='Number of classes')
    parser.add_argument('-s', '--img-size', type=int, default=299, help='Size of Image, num*num')
    parser.add_argument('-l', '--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('-t', '--test-size', type=float, default=0.1, help='Size of test dataset')
    parser.add_argument('-p', '--path', type=str, default='./data', help='train part module is in utils dir')
    parser.add_argument('-d', '--dropout-rate', type=float, default=0.4, help='Dropout rate to tune')
    parser.add_argument('-w', '--weight-decay', type=float, default=1e-8, help='Weight decay constant')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()
    train(args)
    predict(args)
    print("\nDone!!!")