import argparse
import sys
import os


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='nlp embedding')
    args.add_argument('m', '--model', default='w2v' type=str,
                      help='model to train')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-f', '--file', default='D:\\data\\text\\news-articles\\kbanker_articles_subtitles.csv', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='cuda:0', type=str,
                      help='indices of GPUs to enable (default: all)')
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    config = config_parser(args.parse_args())
    main(config)
