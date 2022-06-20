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


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

class Transformer(nn.Module):
    def __init__(self, dim=1024, depth=6, heads=8, dim_head=64, mlp_dim=2048, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.fc = nn.Linear(1024,34)
    def forward(self, x,y):
        x = x.float().unsqueeze(1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.fc(x).squeeze(),y
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1024, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1024),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        #img = img.view(img.size(0), *img_shape)
        return img
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


def train_GAN(dataloader):
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # ----------
    #  Training
    # ----------
    for epoch in range(64):
        for i, (imgs, _) in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 1024))))
            # Generate a batch of images
            gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            #batches_done = epoch * len(dataloader) + i
            #if batches_done % opt.sample_interval == 0:
            #    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    return generator




def random_feature_interpolation(selected_mean, query, k):
    num_q = query.shape[0]
    num_base = selected_mean.shape[0]
    origin = np.random.choice(num_q, 1000)
    target = np.random.choice(num_base, 1000)
    alpha = np.stack([np.random.rand(1000)]*1024, axis=-1)*0.07
    print(query[origin].shape,selected_mean[target].shape)
    generated_feature = query[origin,:] + alpha*selected_mean[target,:]
    #print(generated_feature.shape)
    #sys.exit()
    return generated_feature

def distribution_calibration(query, base_means, base_cov, k,alpha=0.21):
    query = query.numpy()
    dist = []
    k=3
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
    samples = random_feature_interpolation(selected_mean,query,k)
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
        self.fc2 = nn.Linear(256, 34)
        self.fc3=nn.Linear(1024,34)

    def forward(self, x,y):
        y = y.float()
        x=self.fc1(x.float())
        att = self.softmax(self.attention2(self.relu(self.attention(x))))
        return self.fc2(self.relu(self.dropout(x+att*x))), self.fc3(y)
def calculate_samples_per_class(annotation):
    class_index_max = 34
    #print(annotation)
    number_per_class = torch.zeros(int(class_index_max))
    for i in range(class_index_max):
         number_per_class[i] = (annotation == i).sum()
    return number_per_class
def calculate_weights(annotation, weights):
    class_index_max = 34
    #number_per_class = torch.zeros(class_index_max)
    sampler_weight = torch.zeros_like(annotation)
    for i in range(class_index_max):
        mask= annotation == i
        sampler_weight[mask] = weights[i]
    return sampler_weight
if __name__ == '__main__':
    # ---- data loading
    n_runs = 10000
    import FSLTask
    import torch.optim as optim



    rare_data, rare_classes, rich_data, rich_classes, feature_val, annotation_val, feature_test, annotation_test, sample_num_per_class = load_dataset_driveact()
    #acc = sklearn.metrics.top_k_accuracy_score(annotation_val, np.squeeze(feature_val), k=1)
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
    base_means = []
    base_cov = []
    #base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset

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

    for i in range(len(rare_classes)):
        num_sampled = 1000 #(int(torch.max(sample_num_per_class) - sample_num_per_class[rare_classes[i]]))
        count += num_sampled
        mean, cov, samples = distribution_calibration(rare_data[i], base_means, base_cov, k=2)
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
    X_aug_1 = sampled_data#np.concatenate([rare_data, sampled_data])
    Y_aug_1 = sampled_label#np.concatenate([rare_label,sampled_label])
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
        mean, cov, samples = distribution_calibration(rich_data[i], base_means, base_cov, k=2)
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
    X_aug_2 = sampled_data#np.concatenate([rich_data, sampled_data])
    Y_aug_2 = sampled_label#np.concatenate([rich_label,sampled_label])
    X_aug = np.concatenate([X_aug_1, X_aug_2])
    Y_aug = np.concatenate([Y_aug_1, Y_aug_2])
    #X_aug += np.random.normal(0, .1, X_aug.shape)


    #print(torch.Tensor(Y_aug))
    sample_number = calculate_samples_per_class(torch.Tensor(Y_aug))
    #print(sample_number)
    weights = 1/sample_number
    #print(weights)
    sampler_weight = calculate_weights(torch.Tensor(Y_aug), weights)

    # ---- train classifier
    #print(X_aug.shape)
    #if mode == 'train':
    #  samples_weight = torch.from_numpy(np.array([weight[t] for t in dataset.gt_labels]))
    print(sampler_weight)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000993, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    #optimizer = torch.optim.SGD(model.parameters(), 0.001,
    #                            momentum=0.9,
    #                            weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,  eta_min=0, last_epoch=- 1, verbose=False)

    #print(np.min(Y_aug))
    #sys.exit()
    #criterion2 = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=False,)
    for epoch in range(1500):
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
            for sub_epoch in range(10):
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
            predicts,y = model(data.cuda(), data.cuda())
            val_predict.append(predicts.cpu())
    val_predict = torch.cat(val_predict, dim=0).cpu().numpy()
    test_predict = []
    #val_predict = np.argmax(val_predict, axis=-1)
    #print(predicts)
    acc = sklearn.metrics.top_k_accuracy_score(annotation_val, val_predict, k=1)
    print('two-stage calibration eval  ACC : %f'%acc)
    #predicts = model.predict(np.squeeze(feature_test))
    model.eval()
    for step, (data,label) in enumerate(test_dataloader):
        with torch.no_grad():
            predicts,y = model(data.cuda(), data.cuda())
            test_predict.append(predicts.cpu())
    test_predict = torch.cat(test_predict, dim=0).cpu().numpy()
    #print(np.squeeze(feature_test).shape)
    #sys.exit()
    #predicts = model.predict(np.squeeze(feature_val))

    #predicts = np.argmax(predicts, axis=-1)
    #test_predict = np.argmax(test_predict, axis=-1)
    acc = sklearn.metrics.top_k_accuracy_score(annotation_test, test_predict, k=1)

    print('two-stage calibration test  ACC : %f' % acc)
    #filename = 'finalized_model.sav'
    #pickle.dump(model, open(filename, 'wb'))


