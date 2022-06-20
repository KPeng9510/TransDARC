import torch.optim.lr_scheduler
import pickle
import numpy as np
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from FSLTask import load_dataset_driveact
use_gpu = torch.cuda.is_available()
import sklearn
import sys
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
import random
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from sklearn.metrics import confusion_matrix

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
global nc
nc = 34
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def random_feature_interpolation(selected_mean, query, k, num):
    num_q = query.shape[0]
    #print(num_q)
    num_base = selected_mean.shape[0]
    origin = np.random.choice(num_q, num)
    target = np.random.choice(num_base, num)
    alpha = np.stack([np.random.rand(num)]*1024, axis=-1)*0.07
    #print(query[origin].shape,selected_mean[target].shape)
    generated_feature = query[origin,:] + alpha*selected_mean[target,:]
    #print(generated_feature.shape)
    #sys.exit()
    return generated_feature

def distribution_calibration(query, base_means, base_cov, k,alpha, num):
    query = query.numpy()
    dist = []
    k=1
    alpha=0.21
    #print(query.shape)
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    #mean_basics = np.array(base_means)[index]
    #print(mean_basics.shape)
    #sys.exit()
    selected_mean = np.array(base_means)[index]
    mean = np.concatenate([np.array(base_means)[index], np.squeeze(query[np.newaxis, :])])
    #mean = np.squeeze(query[np.newaxis, :])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha
    samples = random_feature_interpolation(selected_mean,query,k, num)
    #print(calibrated_mean)
    #print(calibrated_cov)
    #feature interpolation based feature augmentation

    return calibrated_mean, calibrated_cov, samples

class CustomImageDataset(Dataset):
    def __init__(self, feature, annotation, transform=None, target_transform=None):

        self.feature = feature #.view(-1,34)
        #print(self.feature.shape)
        #sys-exit()
        self.annotations = annotation
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        return self.feature[idx], self.annotations[idx]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.attention = nn.Linear(256,128)
        self.attention2 = nn.Linear(128, 256)
        #self.pool = nn.MaxPool1d(128)
        self.relu = nn.ReLU()
        #self.bn = torch.nn.BatchNorm1d(128)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.softmax = nn.Sigmoid()
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(256,nc)
        self.fc3=nn.Linear(1024,nc)

    def forward(self, x,y):
        y = y.float()
        x=self.fc1(x.float())
        att = self.softmax(self.attention2(self.relu(self.attention(x))))
        return self.fc2(self.relu(self.dropout(x+att*x))),self.fc3(y),

def calculate_samples_per_class(annotation):
    class_index_max = nc
    #print(annotation)
    number_per_class = torch.zeros(int(class_index_max))
    for i in range(class_index_max):
         number_per_class[i] = (annotation == i).sum()
    return number_per_class
def calculate_weights(annotation, weights):
    class_index_max = nc
    #number_per_class = torch.zeros(class_index_max)
    sampler_weight = torch.zeros_like(annotation)
    for i in range(class_index_max):
        mask= annotation == i
        sampler_weight[mask] = weights[i]
    return sampler_weight
def generate_train(rare_data, rare_classes, rich_data, rich_classes,sample_num_per_class):
    base_means = []
    base_cov = []
    for key in range(len(rich_data)):
        feature = np.array(rich_data[key])
        print(feature.shape)
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_means.append(mean)
        base_cov.append(cov)

    # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))

    #support_data = ndatas[i][:n_lsamples].numpy()
    #support_label = labels[i][:n_lsamples].numpy()
    #query_data = ndatas[i][n_lsamples:].numpy()
    #query_label = labels[i][n_lsamples:].numpy()
    # ---- Tukey's transform
    beta = 0.5
    #support_data = np.power(support_data[:, ] ,beta)
    #query_data = np.power(query_data[:, ] ,beta)
    # ---- cross distribution calibration for rare classes
    sampled_data = []
    sampled_label = []
    count = 0
    np.set_printoptions(threshold=sys.maxsize)
    #for i in range(len(rare_classes)):
    #    print(rare_data[i].shape)
    for i in range(len(rare_classes)):
        print(sample_num_per_class[i])

        #if sample_num_per_class[rare_classes[i]] == 0:
        #    continue
        num_sampled = 1000 #(int(torch.max(sample_num_per_class) - sample_num_per_class[rare_classes[i]]))
        count += num_sampled
        #print(rare_data[i].shape)
        mean, cov, samples = distribution_calibration(rare_data[i], base_means, base_cov, 2, 0.21, 1000)
        #print(num_sampled)
        #print(samples.shape)
        #print(cov)
        sampled_data.append(samples)
        #sampled_data.append(np.random.multivariate_normal(list(mean), list(cov), num_sampled, 'warn'))
        #print(np.mean(sampled_data[i], axis=0))
        #print(np.max(sampled_data[i], axis=0))
        #val_data = feature_val[annotation_val==rare_classes[i]].numpy()
        #print(val_data)
        #print(np.mean(val_data, axis=0)-np.mean(sampled_data[i], axis=0))
        sampled_label.extend([rare_classes[i]]*num_sampled)
        #sampled_label.extend([rare_classes[i]] * int(sample_num_per_class[rare_classes[i]]))
        #sys.exit()
    #sys.exit()
    sampled_data = np.concatenate(sampled_data).reshape(count, 1024)
    rare_data = np.concatenate(rare_data, axis=0)
    rare_label = []#torch.zeros(rare_data.shape[0])
    for i in range(len(rare_classes)):
        rare_label.extend([rare_classes[i]] * int(sample_num_per_class[rare_classes[i]]))
    rare_label = np.array(rare_label)
    X_aug_1 = sampled_data #np.concatenate([rare_data, sampled_data])
    Y_aug_1 = sampled_label #np.concatenate([rare_label,sampled_label])
    #print(X_aug_1.shape)
    #print(Y_aug_1.shape)

    # ---- self distribution calibration for rich classes
    #num_sampled = int(750 / n_shot)
    sampled_data = []
    sampled_label = []
    count = 0

    for i in range(len(rich_classes)):
        num_sampled = 1000 #int(torch.max(sample_num_per_class) - sample_num_per_class[rich_classes[i]])
        #if sample_num_per_class>500:
        #    continue
        count += num_sampled
        #print(rich_classes[i])
        #mean, conv, samples = base_means[i], base_cov[i]
        #print(samples.shape)
        mean, cov, samples = distribution_calibration(rich_data[i], base_means, base_cov, 2, 0.21, 1000)
        #sampled_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=num_sampled))
        sampled_data.append(samples)
        sampled_label.extend([rich_classes[i]] * num_sampled)
        #sampled_label.extend([rich_classes[i]] * int(sample_num_per_class[rich_classes[i]]))
    sampled_data = np.concatenate(sampled_data).reshape(count, 1024)
    rich_label = []#torch.zeros(rare_data.shape[0])
    for i in range(len(rich_classes)):
        rich_label.extend([rich_classes[i]] * int(sample_num_per_class[rich_classes[i]]))
    rare_label = np.array(rich_label)
    rich_data = np.concatenate(rich_data, axis=0)
    X_aug_2 = sampled_data #rich_data #sampled_data #np.concatenate([rich_data, sampled_data])
    Y_aug_2 = sampled_label #rich_label#sampled_label #np.concatenate([rich_label,sampled_label])
    X_aug = np.concatenate([X_aug_1, X_aug_2])
    Y_aug = np.concatenate([Y_aug_1, Y_aug_2])
    #X_aug += np.random.normal(0, .1, X_aug.shape)
    return X_aug, Y_aug

if __name__ == '__main__':
    # ---- data loading
    n_runs = 10000
    import FSLTask
    import torch.optim as optim



    rare_data, rare_classes, rich_data, rich_classes, feature_val, annotation_val, feature_test, annotation_test, sample_num_per_class,rare_data_aug, rare_classes_aug, rich_data_aug, rich_classes_aug,sample_num_per_class_aug = load_dataset_driveact()
    #acc = sklearn.metrics.top_k_accuracy_score(annotation_test, np.squeeze(feature_test), k=1)
    #print(acc)
    #noise_mean = np.zeros(34, 1024)

    #print(len(rare_data)+len(rich_data))
    #print(len(annotation_test))
    #sys.exit()
    #print(len(annotation_test))
    #length= rare_classes.shape[0]+rich_data.shape[]
    #cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    #FSLTask.loadDataSet(dataset)
    #FSLTask.setRandomStates(cfg)
    #rich_datas, rare_data = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    #ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    #labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,                                                                                        n_samples)
    # ---- Base class statistics

    #base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset

    X_aug, Y_aug = generate_train(rare_data,rare_classes,rich_data,rich_classes,sample_num_per_class)
    X_aug2, Y_aug2 = generate_train(rare_data_aug,rare_classes_aug,rich_data_aug,rich_classes_aug,sample_num_per_class_aug)
    #print(torch.Tensor(Y_aug))
    X_aug = np.concatenate([X_aug, X_aug2])
    Y_aug = np.concatenate([Y_aug, Y_aug2])
    sample_number = calculate_samples_per_class(torch.Tensor(Y_aug))
    #print(sample_number)
    weights = 1/sample_number
    #print(weights)
    sampler_weight = calculate_weights(torch.Tensor(Y_aug), weights)

    # ---- train classifier
    #print(X_aug.shape)
    #if mode == 'train':
    #  samples_weight = torch.from_numpy(np.array([weight[t] for t in dataset.gt_labels]))
    #print(sampler_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sampler_weight,3000)


    dataset_train = CustomImageDataset(X_aug, Y_aug)
    #train_GAN(DataLoader(dataset_train, batch_size=256, sampler=sampler))
    dataset_val = CustomImageDataset(np.squeeze(feature_val), annotation_val)
    dataset_test = CustomImageDataset(np.squeeze(feature_test), annotation_test)



    train_dataloader = DataLoader(dataset_train, batch_size=256, sampler=sampler)


    infer_dataloader = DataLoader(dataset_train, batch_size=256, shuffle=False)
    test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False)

    #model = LogisticRegression(max_iter=10000,verbose=10).fit(X=X_aug, y=Y_aug)
    model = Net()
    #resume = '/cvhci/temp/kpeng/driveact/models_swin_base/best_top1_acc_epoch_24.pth'
    #checkpoint = torch.load(resume)
    #print(checkpoint['state_dict']['cls_head.fc_cls.weight'])
    #print(checkpoint['state_dict']['cls_head.fc_cls.bias'])
    #model.fc1.weight.data = checkpoint['state_dict']['cls_head.fc_cls.weight']
    #model.fc1.bias.data = checkpoint['state_dict']['cls_head.fc_cls.bias']
    #sys.exit()
    model=model.cuda()
    criterion = nn.CrossEntropyLoss(reduce='None',reduction='mean')
    criterion2 = nn.CrossEntropyLoss(reduce=False,reduction='none')
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000993, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    #optimizer = torch.optim.SGD(model.parameters(), 0.001,
    #                            momentum=0.9,
    #                            weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,  eta_min=0, last_epoch=- 1, verbose=False)

    #print(np.min(Y_aug))
    #sys.exit()
    #criterion2 = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=False,)
    for epoch in range(3000):
        hard_samples = []
        model.train()
        for step, (data,label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicts,y = model(data.cuda(), data.cuda())

            loss = criterion(predicts, label.cuda()) #+ criterion2(predicts,y, torch.ones(y.size()[0]).cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
        if (epoch > 50) and (epoch%30 == 0):
            model.eval()
            for index, (data, label) in enumerate(infer_dataloader):
                with torch.no_grad():
                    #label = torch.Tensor(label).cuda().double()
                    predicts,y = model(data.cuda(), data.cuda())
                    #print(predicts.size())
                    difficulty = criterion2(predicts, label.cuda())
                    #print(difficulty)
                    hard_samples.append(difficulty.data)
            #print(hard_samples)
            difficulty = torch.cat(hard_samples, dim=0)
            threshold = 1.2* torch.mean(difficulty)
            mask = (difficulty>threshold).cpu().numpy()
            hard_set = X_aug[mask]
            hard_label = Y_aug[mask]
            hard_dataset = CustomImageDataset(hard_set, hard_label)

            hard_train_dataloader = DataLoader(hard_dataset, batch_size=256, shuffle=True)
            for sub_epoch in range(1):
                model.train()
                for step, (data,label) in enumerate(hard_train_dataloader):
                    optimizer.zero_grad()
                    predicts,y = model(data.cuda(), data.cuda())
                    loss = 3*criterion(predicts, label.cuda()) #+ criterion2(predicts,y, torch.ones(y.size()[0]).cuda())
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()
                print(epoch, 'hard_loss', loss)
        print(epoch, 'loss', loss)

    val_predict = []
    model.eval()
    for step, (data,label) in enumerate(val_dataloader):
        with torch.no_grad():
            #data = torch.nn.functional.normalize(data, dim=-1)

            predicts,y = model(data.cuda(), data.cuda())
            val_predict.append(predicts.cpu())
    val_predict = torch.cat(val_predict, dim=0).cpu().numpy()
    test_predict = []
    #val_predict = np.argmax(val_predict, axis=-1)
    #print(predicts)
    acc = sklearn.metrics.top_k_accuracy_score(annotation_val, val_predict, k=1)
    f = open('/cvhci/data/activity/kpeng/ts_val_midlevel_predict_split0.pkl', 'wb')
    pickle.dump(val_predict,f)
    f.close()
    f = open('/cvhci/data/activity/kpeng/ts_val_midlevel_label_split0.pkl', 'wb')
    pickle.dump(annotation_val,f)
    f.close()
    print('two-stage calibration eval  ACC : %f'%acc)
    #predicts = model.predict(np.squeeze(feature_test))
    cm = confusion_matrix(annotation_val, np.argmax(val_predict, axis=-1))
    f = open("/cvhci/data/activity/Drive&Act/kunyu/annotation_list.pkl", 'rb')
    annotation = []
    class_index = pickle.load(f)
    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}
    # Calculate the accuracy for each one of our classes
    for idx, cls in enumerate(range(nc)):
        # True negatives are all the samples that are not our current GT class (not the current row)
        # and were not predicted as the current class (not the current column)
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        # True positives are all the samples of our current GT class that were predicted as such
        true_positives = cm[idx, idx]
        # The accuracy for the current class is ratio between correct predictions to all predictions
        per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)
        #print(class_index[idx], 'val_accuracy', per_class_accuracies[cls])
    model.eval()


    for step, (data,label) in enumerate(test_dataloader):
        with torch.no_grad():
            data = torch.nn.functional.normalize(data, dim=-1)
            predicts,y = model(data.cuda(), data.cuda())
            test_predict.append(predicts.cpu())
    test_predict = torch.cat(test_predict, dim=0).cpu().numpy()
    #print(np.squeeze(feature_test).shape)
    #sys.exit()
    #predicts = model.predict(np.squeeze(feature_val))
    #predicts = np.argmax(predicts, axis=-1)
    #test_predict = np.argmax(test_predict, axis=-1)
    acc = sklearn.metrics.top_k_accuracy_score(annotation_test, test_predict, k=1)
    f = open('/cvhci/data/activity/kpeng/ts_test_midlevel_predict_split0.pkl', 'wb')
    pickle.dump(test_predict,f)
    f.close()
    f = open('/cvhci/data/activity/kpeng/ts_test_midlevel_label_split0.pkl', 'wb')
    pickle.dump(annotation_test,f)
    f.close()
    print('two-stage calibration test  ACC : %f' % acc)
    #for i in range(34):
    #    mask = annotation_test == i
    #    acc = sklearn.metrics.top_k_accuracy_score(torch.argmax(annotation_test[mask], dim=-1), test_predict[mask], k=1)
    #    print('class', class_index[i], 'accuracy:', acc)
    #filename = 'finalized_model.sav'
    #pickle.dump(model, open(filename, 'wb'))
    # Get the confusion matrix
    cm = confusion_matrix(annotation_test, np.argmax(test_predict, axis=-1))
    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}
    # Calculate the accuracy for each one of our classes
    for idx, cls in enumerate(range(nc)):
        # True negatives are all the samples that are not our current GT class (not the current row)
        # and were not predicted as the current class (not the current column)
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        # True positives are all the samples of our current GT class that were predicted as such
        true_positives = cm[idx, idx]
        # The accuracy for the current class is ratio between correct predictions to all predictions
        per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)
        #print(class_index[idx], 'test_accuracy', per_class_accuracies[cls])
    cm = confusion_matrix(annotation_test, np.argmax(feature_test, axis=-1))
    # We will store the results in a dictionary for easy access later
    per_class_accuracies = {}
    # Calculate the accuracy for each one of our classes
    for idx, cls in enumerate(range(nc)):
        # True negatives are all the samples that are not our current GT class (not the current row)
        # and were not predicted as the current class (not the current column)
        true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
        # True positives are all the samples of our current GT class that were predicted as such
        true_positives = cm[idx, idx]
        # The accuracy for the current class is ratio between correct predictions to all predictions
        per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)
        #print(class_index[idx], 'test_accuracy', per_class_accuracies[cls])



