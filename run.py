#-*- coding:utf-8 -*-
import os
import sys
import warnings
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings('ignore')

# "/home/mist/wang/sentiment/pretrained/chinese_roberta_wwm_ext_pytorch/"
bert_path_map = {'albert_xxlarge_zh': "voidful/albert_chinese_xxlarge",
                 'chinese_roberta_wwm_ext_pytorch': "../sentiment/pretrained/chinese_roberta_wwm_ext_pytorch/",
                 'debert': "WENGSYX/Deberta-Chinese-Large"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="Autobert", required=False)
    parser.add_argument('--train_input_path', type=str, default="../data/nlp_data/new_train.json",
                        required=False,help="已经生成的训练集文件位置")
    parser.add_argument('--test_input_path', type=str, required=False, default="../data/nlp_data/new_test.json",help="已经生成的测试集文件位置")
    parser.add_argument('--save_model_path', type=str, required=False, default="../sentiment/model_save",help="模型保存位置")
    parser.add_argument('--result_path', type=str, required=False, default="../sentiment/result/section2.txt",help="结果存放位置")
    parser.add_argument('--pretrained_path', type=str, required=False,
                        default="microsoft/deberta-base")
    parser.add_argument('--checkpoint_path', type=str, required=False, default=None,help="继续训练")
    parser.add_argument('--epochs', type=int, default=10, required=False)
    parser.add_argument('--max_sequence_input', type=int, default=256, required=False)
    parser.add_argument('--batch_size', type=int, default=16, required=False) #64
    parser.add_argument('--save_state', action='store_true', required=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001, required=False)
    parser.add_argument('--tokenizer_path', default='../sentiment/pretrain_model/debert/vocab.txt', type=str,
                        required=False)
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--ratio', default=0.8, type=float, required=False, help='trainset ratio')

    args = parser.parse_args()
    if args.model_type == "Autobert":
        from sentiment import train
        train.main(args)


if __name__ == '__main__':
    main()
