import os
import pickle
import numpy as np
import torch
import sys
from pprint import pprint
# from tqdm import tqdm

# ========================================================
#   Usefull paths
_datasetFeaturesFiles = {"train": "/cvhci/data/activity/kpeng/logits_split0_chunk90_swin_base_last_logits768_train.pkl",
                         "eval": "/cvhci/data/activity/kpeng/logits_split0_chunk90_swin_base_last_logits768_val.pkl",
                         "test": "/cvhci/data/activity/kpeng/logits_split0_chunk90_swin_base_last_logits768_test.pkl"}
_cacheDir = "./cache"
_maxRuns = 10000
_min_examples = -1

# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None



def load_label_feature(item):
    f = open("/cvhci/data/activity/Drive&Act/kunyu/annotation_list.pkl", 'rb')
    annotation = []
    class_index = pickle.load(f)
    f.close()
    infos = item.keys()
    for info in infos:
        info = ''.join([item[0] for item in list(info)])
        #print(info)
        activity = info.split(',')[-2]
        label = class_index.index(activity)
        annotation.append(label)
    features = item.values()
    #print(features)
    feature = [term for term in features]
    #print(feature)
    return feature, annotation

def _load_pickle(file_train,file_eval,file_test):
    dataset = dict()
    with open(file_train, 'rb') as f:
        data = pickle.load(f)
        feature, annotation = load_label_feature(data)
        #labels = [np.full(shape=len(data[key]), fill_value=key)
        #          for key in data]
        #data = [features for key in data for features in data[key]]
        dataset['data_train'] = torch.FloatTensor(np.stack(feature, axis=0))
        #print(dataset['data_train'].size())
        dataset['labels_train'] = torch.LongTensor(np.stack(annotation, axis=0))
    with open(file_eval, 'rb') as f:
        data = pickle.load(f)
        feature, annotation = load_label_feature(data)
        #labels = [np.full(shape=len(data[key]), fill_value=key)
        #          for key in data]
        #data = [features for key in data for features in data[key]]

        dataset['data_eval'] = torch.FloatTensor(np.stack(feature, axis=0))
        dataset['labels_eval'] = torch.LongTensor(np.stack(annotation, axis=0))
        #print(dataset['labels_eval'])
        #sys.exit()
    with open(file_test, 'rb') as f:
        data = pickle.load(f)
        feature, annotation = load_label_feature(data)
        #labels = [np.full(shape=len(data[key]), fill_value=key)
        #          for key in data]
        #data = [features for key in data for features in data[key]]
        dataset['data_test'] = torch.FloatTensor(np.stack(feature, axis=0))
        dataset['labels_test'] = torch.LongTensor(np.stack(annotation,axis=0))
    return dataset

def calculate_samples_per_class(annotation):
    class_index_max = torch.max(annotation)+1
    number_per_class = torch.zeros(class_index_max)
    for i in range(class_index_max):
         number_per_class[i] = (annotation == i).sum()
    return number_per_class
def arrange_dataset(data, annotation):
    class_index_max = torch.max(annotation)+1
    arranged_dataset = []
    for i in range(class_index_max):
        mask = annotation == i
        arranged_dataset.append(data[mask,:,:])
    return arranged_dataset
def rare_class_selection(data, sam_number):
    #print(data.size())
    rare_class_threshold = 100
    annotation = torch.arange(34)
    #print(sam_number)
    rare_mask = (sam_number < rare_class_threshold).bool()
    rich_mask = sam_number >= rare_class_threshold
    rare_mask_list = [int(item) for item in annotation[rare_mask].tolist()]
    #print(rare_mask_list)
    #print(data)
    rare_data = [data[i].squeeze() for i in rare_mask_list]
    rare_classes = annotation[rare_mask]
    #rich_data = data[rich_mask, :,:]
    rich_classes = annotation[rich_mask]
    rich_mask_list = [int(item) for item in annotation[rich_mask].tolist()]
    #print(rich_mask_list)
    # print(data)
    rich_data = [data[i].squeeze() for i in rich_mask_list]
    return rare_data, rare_classes, rich_data, rich_classes
def load_dataset_driveact():
    train_path = _datasetFeaturesFiles['train']
    val_path= _datasetFeaturesFiles['eval']
    test_path = _datasetFeaturesFiles['test']
    #load_train_dataset

    dataset = _load_pickle(train_path, val_path, test_path)
    feature_train = dataset['data_train']
    annotation_train = dataset['labels_train']
    feature_val, annotation_val = dataset['data_eval'], dataset['labels_eval']
    feature_test, annotation_test = dataset['data_test'], dataset['labels_test']
    sample_per_class_train = calculate_samples_per_class(annotation_train)
    arrange_dataset_train = arrange_dataset(feature_train, annotation_train)
    rare_data, rare_classes, rich_data, rich_classes = rare_class_selection(arrange_dataset_train, sample_per_class_train)

    return rare_data, rare_classes, rich_data, rich_classes, feature_val,annotation_val,feature_test, annotation_test, sample_per_class_train


# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None


def loadDataSet(dsname):
    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName, data, labels, _randStates, _rsCfg, _min_examples
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    dataset = _load_pickle(_datasetFeaturesFiles[dsname])

    # Computing the number of items per class in the dataset
    _min_examples = dataset["labels"].shape[0]
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
                          [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]))


def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_indices = np.arange(_min_examples)
    dataset = None
    if generate:
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        if generate:
            dataset[i] = data[classes[i], shuffle_indices,
                              :][:cfg['shot']+cfg['queries']]

    return dataset


def ClassesInRun(iRun, cfg):
    global _randStates, data
    np.random.set_state(_randStates[iRun])
    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes


def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return
    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


def GenerateRunSet(start=None, end=None, cfg=None):
    global dataset, _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for iRun in range(end-start):
        dataset[iRun] = GenerateRun(start+iRun, cfg)

    return dataset


# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    loadDataSet('miniimagenet')

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)

    run10 = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = GenerateRunSet(start=2, end=12, cfg=cfg)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())
