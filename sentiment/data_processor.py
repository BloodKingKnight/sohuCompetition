import json
import warnings
from typing import List
warnings.filterwarnings("ignore")
# nlp训练集路径
TRAIN_FILE_NAME = "../data/nlp_data/train.json"
TEST_FILE_NAME = "../data/nlp_data/test.json"
NEW_TRAIN_FILE = "../data/nlp_data/new_train.json"
NEW_TEST_FILE = "../data/nlp_data/new_test.json"

"""
    加载处理json文件，返回处理好的字典
    params: 
    
        file_path(str): json文件路径
        
    returns:
    
        data_content(dict): id映射content的字典
        
        data_entity(dict): id映射entity的字典
        
"""
def read_train_file(file_path: str) -> (List[str], List[int], dict):
    corpus = []
    labels = []
    entitys = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line.strip())
            raw_contents = tmp['content']
            raw_entitys = tmp['entity']
            label = int(tmp["label"])
            if label == -2:
                label = 4
            elif label == -1:
                label = 3
            for entity in [raw_entitys]:
                text = raw_contents.strip()
                corpus.append(text)
                entitys.append(entity)
                labels.append(label)
    # print(corpus[0])
    # print(labels[0])
    # print(entitys[0])
    assert len(corpus) == len(labels) == len(entitys)
    return corpus, labels, entitys


    # neutral: 0.715359146019265, positive: 0.15830347923463625, negative: 0.12633737474609877
    # 数据失衡
    # neutral = 0
    # positive = 0
    # negative = 0
    #
    # for k, v in data_entity.items():
    #     for entity, sentiment in v.items():
    #         if sentiment > 0:
    #             positive += 1
    #         elif sentiment < 0:
    #             negative += 1
    #         else:
    #             neutral += 1
    # emotion = neutral + positive + negative
    # print(f"neutral: {neutral / emotion}, positive: {positive / emotion}, negative: {negative / emotion}")
    #

    # 89195条数据

def read_test_file(file_path: str):
    ids = []
    corpus = []
    entities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp = json.loads(line.strip())
            raw_id = tmp['id']
            raw_contents = tmp['content']
            raw_entities = tmp['entity']
            for entity in [raw_entities]:
                text = raw_contents.strip()
                corpus.append(text)
                ids.append(raw_id)
                entities.append(entity)
    assert len(corpus) == len(entities) == len(ids)
    return corpus, entities, ids

def reconstruct_data_piece(id: int, content: str, entity: str, label: int, is_train: bool) -> dict:
    if is_train:
        return {
            "id": id,
            "content": content,
            "entity": entity,
            "label": label
        }
    else:
        return {
            "id": id,
            "content": content,
            "entity": entity,
        }

def generate_new_train_data(source_file_path: str, target_file_path: str) -> None:
    with open(source_file_path, "r", encoding="utf-8") as source_file:
        lines = source_file.readlines()
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            for line in lines:
                sourceData = json.loads(line.strip())
                contents = sourceData["content"]
                entities = sourceData["entity"]
                data_ids = sourceData["id"]
                for entity, label in entities.items():
                    target_data = reconstruct_data_piece(id=data_ids, content=contents,
                                                         entity=entity, label=int(label), is_train=True)
                    target_file.write(json.dumps(target_data, ensure_ascii=False) + "\n")

def generate_new_test_data(source_file_path: str, target_file_path: str) -> None:
    with open(source_file_path, "r", encoding="utf-8") as source_file:
        lines = source_file.readlines()
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            for line in lines:
                sourceData = json.loads(line.strip())
                contents = sourceData["content"]
                entities = sourceData["entity"]
                data_ids = sourceData["id"]
                for entity in entities:
                    target_data = reconstruct_data_piece(id=data_ids, content=contents,
                                                         entity=entity, label=0, is_train=False)
                    target_file.write(json.dumps(target_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generate_new_test_data(TEST_FILE_NAME, NEW_TEST_FILE)