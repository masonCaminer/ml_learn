# coding = utf-8
import re


def text_parse(big_string):
    """
    接收一个大字符串并将其解析为字符串列表
    :param big_string:
    :return:
    """
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    list_of_tokens = re.split(r'\W*', big_string)
    # 除了单个字母，例如大写的I，其它单词变成小写
    return [tok for tok in list_of_tokens if len(tok) > 2]


def create_vocab_list(data_set):
    """
    创建词向量表
    :param data_set:
    :return:
    """
    # 创建一个空的不重复列表
    vocab_set = set([])
    for document in data_set:
        # 取并集
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


if __name__ == '__main__':
    doc_list = []
    class_list = []
    # 遍历25个文件
    for i in range(1, 26):
        if i in (6, 17, 22,23):
            continue
        # 读取垃圾文件
        word_list = text_parse(open('email/spam/%d.txt' % i, 'r', encoding='utf-8').read())
        doc_list.append(word_list)
        # 垃圾文件标记为1
        class_list.append(1)
        # 读取非垃圾文件
        word_list = text_parse(open('email/ham/%d.txt' % i, 'r', encoding='utf-8').read())
        doc_list.append(word_list)
        class_list.append(0)
    # 创建词汇表，不重复
