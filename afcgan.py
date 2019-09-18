import argparse
import os
import random
import sys
import time
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import classifier
import classifier2
import model
import soft_cls
import util
import torch
import numpy as np

def GetNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def loadPretrainedMain(netS, savePost):
    print('Loading pretrained Mainnet......')
    path = './pretrain/'
    netS.load_state_dict( torch.load( path+savePost, map_location='cuda:0' ) )


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/home/poxiaoge/Documents/dataset/ZSL', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=3)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')

parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')

parser.add_argument('--loss_syn_num', type=int, default=30, help='G learning rate')
parser.add_argument('--cyc_seen_weight', type=float, default=0.01, help='weight of the seen class cycle loss')
parser.add_argument('--cyc_unseen_weight', type=float, default=0.01, help='weight of the unseen class cycle loss')

parser.add_argument('--dm_seen_weight', type=float, default=0.01, help='weight of the seen class cycle loss')
parser.add_argument('--dm_unseen_weight', type=float, default=0.01, help='weight of the unseen class cycle loss')
parser.add_argument('--dm_weight', type=float, default=0.01, help='weight of the unseen class cycle loss')

parser.add_argument('--cls_syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--cls_batch_size', type=int, default=5, help='G learning rate')
parser.add_argument('--f_hid', type=int, default=4096, help='forward hidden units')

parser.add_argument('--new_lr', type=int, default=0, help='forward hidden units')
parser.add_argument('--ensemble_ratio', type=float, default=0.8, help='forward hidden units')
parser.add_argument('--use_pretrain_s', type=int, default=1, help='forward hidden units')




print(GetNowTime())
print('Begin run!!!')
since = time.time()

opt = parser.parse_args()
sys.stdout.flush()


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

data = util.DATA_LOADER(opt)
print("Training Samples: ", data.ntrain)

netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))


if opt.dataset == 'CUB':
    opt.f_hid = 7000
if opt.dataset == 'FLO':
    opt.f_hid = 7000
if opt.dataset == 'SUN':
    opt.f_hid = 7000
if opt.dataset == 'AWA1':
    opt.f_hid = 3072
if opt.dataset == 'APY':
    opt.f_hid = 6144

netS = model.MLP_V2S(opt)
dm_classifier = model.DomainClassifier(opt.resSize)

cls_criterion = nn.NLLLoss()
reg_criterion = nn.MSELoss()
logsoftmax = nn.LogSoftmax(dim=1)
cnp_criterion = nn.CrossEntropyLoss()


input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)

input_res2 = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att2 = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label2 = torch.LongTensor(opt.batch_size)

input_res3 = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att3 = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label3 = torch.LongTensor(opt.batch_size)


if opt.cuda:
    netD.cuda()
    netG.cuda()
    netS.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    reg_criterion.cuda()
    cnp_criterion.cuda()
    logsoftmax.cuda()
    input_label = input_label.cuda()
    dm_classifier.cuda()

    input_res2 = input_res2.cuda()
    input_att2 = input_att2.cuda()
    input_label2 = input_label2.cuda()

    input_res3 = input_res3.cuda()
    input_att3 = input_att3.cuda()
    input_label3 = input_label3.cuda()

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))
def sample2():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res2.copy_(batch_feature)
    input_att2.copy_(batch_att)
    input_label2.copy_(util.map_label(batch_label, data.seenclasses))
def sample3():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res3.copy_(batch_feature)
    input_att3.copy_(batch_att)
    input_label3.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):  # 每个类都生成num个
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label

def generate_syn_feature_with_grad(netG, classes, attribute, num):
    nclass = classes.size(0)
    # syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(nclass * num, opt.attSize)
    syn_noise = torch.FloatTensor(nclass * num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        syn_label = syn_label.cuda()
    syn_noise.normal_(0, 1)
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.narrow(0, i * num, num).copy_(iclass_att.repeat(num, 1))
        syn_label.narrow(0, i * num, num).fill_(iclass)
    syn_feature = netG(Variable(syn_noise), Variable(syn_att))
    return syn_feature, syn_label.cpu()

d_lr = opt.lr
g_lr = opt.lr
if opt.new_lr == 1:
    d_lr = 1e-3
    g_lr = 1e-4

optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(opt.beta1, 0.999))

optimizerS = optim.Adam(netS.parameters(), lr=1e-4, betas=(opt.beta1, 0.999))
optimizer_dm = optim.Adam(dm_classifier.parameters(), lr=1e-4, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


pretrain_cls = classifier.CLASSIFIER(data, data.train_feature, util.map_label(data.train_label, data.seenclasses),
                                     data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 4096,
                                     opt.pretrain_classifier)

def getTrainSeenAcc():
    fake_train_attr = netS(Variable(data.train_feature.cuda(), volatile=True))
    dist = pairwise_distances(fake_train_attr.data, data.attribute[data.seenclasses].cuda())  # range 150
    pred_idx = torch.min(dist, 1)[1]
    pred = data.seenclasses[pred_idx.cpu()]
    acc = sum(pred == data.train_label) / data.train_label.size()[0]
    print('Train Seen Acc: {:.2f}%'.format(acc * 100))

def getTestUnseenAcc():
    fake_unseen_attr = netS(Variable(data.test_unseen_feature.cuda(), volatile=True))
    dist = pairwise_distances(fake_unseen_attr.data, data.attribute[data.unseenclasses].cuda())  # range 50
    pred_idx = torch.min(dist, 1)[1]  # relative pred
    pred = data.unseenclasses[pred_idx.cpu()]  # map relative pred to absolute pred
    acc = sum(pred == data.test_unseen_label) / data.test_unseen_label.size()[0]
    print('Test Unseen Acc: {:.2f}%'.format(acc * 100))
    return logsoftmax(Variable(dist.cuda())).data

def getTestAllAcc():
    fake_unseen_attr = netS(Variable(data.test_unseen_feature.cuda(), volatile=True))
    dist1 = pairwise_distances(fake_unseen_attr.data, data.attribute.cuda())  # range 200
    pred_idx = torch.min(dist1, 1)[1]  # absolute pred
    acc_unseen = sum(pred_idx.cpu() == data.test_unseen_label) / data.test_unseen_label.size()[0]

    fake_seen_attr = netS(Variable(data.test_seen_feature.cuda(), volatile=True))
    dist2 = pairwise_distances(fake_seen_attr.data, data.attribute.cuda())  # range 200
    pred_idx = torch.min(dist2, 1)[1]  # absolute pred
    acc_seen = sum(pred_idx.cpu() == data.test_seen_label) / data.test_seen_label.size()[0]

    if (acc_seen == 0) or (acc_unseen == 0):
        H = 0
    else:
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
    print('Forward Seen:{:.2f}%, Unseen:{:.2f}%, H:{:.2f}%'.format(acc_seen * 100, acc_unseen * 100, H * 100))
    return logsoftmax(Variable(dist1.cuda())).data, logsoftmax(Variable(dist2.cuda())).data

if opt.use_pretrain_s == 1:
    modelStr = {'CUB':'netS_CUB_Acc415900_03_15_11_23.pth','FLO':'netS_FLO_Acc269300_03_15_10_26.pth','SUN':'netS_SUN_Acc457600_03_15_10_44.pth',
            'AWA1':'netS_AWA1_Acc450100_03_15_10_58.pth','APY':'netS_APY_Acc203100_03_15_11_08.pth'}
    loadPretrainedMain(netS,modelStr[opt.dataset])
else:
    netS.train()
    for epoch in range(50):
        for i in range(0, data.ntrain, opt.batch_size):
            optimizerS.zero_grad()
            sample()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            pred = netS(input_resv)
            loss = reg_criterion(pred, input_attv)
            loss.backward()
            optimizerS.step()
        print(epoch)
    for p in netS.parameters():
        p.requires_grad = False
    netS.eval()


if opt.gzsl:
    opt.gzsl_unseen_output, opt.gzsl_seen_output = getTestAllAcc()
    opt.fake_test_seen_attr = netS(Variable(data.test_seen_feature.cuda(), volatile=True)).data
    opt.fake_test_unseen_attr = netS(Variable(data.test_unseen_feature.cuda(), volatile=True)).data
else:
    opt.zsl_unseen_output = getTestUnseenAcc()
    opt.fake_test_attr = netS(Variable(data.test_unseen_feature.cuda(), volatile=True)).data


for p in pretrain_cls.model.parameters():  # set requires_grad to False
    p.requires_grad = False
pretrain_cls.model.eval()

for epoch in range(opt.nepoch):
    print('EP[%d/%d]****************************************************************************************************************' % (epoch, opt.nepoch))

    for i in range(0, data.ntrain, opt.batch_size):
        for p in netD.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = False

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optimizerD.step()

        for p in netD.parameters():
            p.requires_grad = False
        for q in dm_classifier.parameters():
            q.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = False

        optimizer_dm.zero_grad()
        sample2()
        input_resv2 = Variable(input_res2)
        fake_unseen_feature1, fake_unseen_label1 = generate_syn_feature(netG, data.unseenclasses,data.attribute, opt.loss_syn_num)  # 每个类生成2个sample;31x2=62
        src_label_dm = torch.ones(input_label2.size()).long().cuda()
        tgt_label_dm = torch.zeros(fake_unseen_label1.size()).long().cuda()
        src_label_dm = Variable(src_label_dm)
        tgt_label_dm = Variable(tgt_label_dm)
        src_output_dm = dm_classifier(input_resv2)
        tgt_output_dm = dm_classifier(Variable(fake_unseen_feature1.cuda()))
        loss_dm_src = cnp_criterion(src_output_dm, src_label_dm)
        loss_dm_tgt = cnp_criterion(tgt_output_dm, tgt_label_dm)
        loss_dm = opt.dm_seen_weight * loss_dm_src + opt.dm_unseen_weight * loss_dm_tgt
        loss_dm.backward()
        optimizer_dm.step()

        for q in dm_classifier.parameters():
            q.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = True
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label))

        unseen_feature, unseen_label = generate_syn_feature_with_grad(netG, data.unseenclasses, data.attribute,opt.loss_syn_num)
        unseen_attr = Variable(data.attribute[unseen_label].cuda())
        seen_feature, seen_label = generate_syn_feature_with_grad(netG, data.seenclasses, data.attribute,opt.loss_syn_num)
        seen_attr = Variable(data.attribute[seen_label].cuda())
        r_errG_seen = reg_criterion(netS(seen_feature), seen_attr)
        r_errG_unseen = reg_criterion(netS(unseen_feature), unseen_attr)

        errG = G_cost + opt.cls_weight * c_errG + opt.cyc_seen_weight * r_errG_seen + opt.cyc_unseen_weight * r_errG_unseen
        errG.backward()

        fake_unseen_feature2, fake_unseen_label2 = generate_syn_feature_with_grad(netG, data.unseenclasses,data.attribute,opt.loss_syn_num)
        sample3()
        input_resv3 = Variable(input_res3)
        feature_concat = torch.cat((input_resv3, fake_unseen_feature2), 0)
        output_dm_conf = dm_classifier(feature_concat)
        output_dm_conf = F.softmax(output_dm_conf, dim=1)
        uni_distrib = torch.FloatTensor(output_dm_conf.size()).uniform_(0, 1)
        uni_distrib = uni_distrib.cuda()
        uni_distrib = Variable(uni_distrib)
        loss_conf = -opt.dm_weight * (torch.sum(uni_distrib * torch.log(output_dm_conf))) / float(output_dm_conf.size(0))
        loss_conf.backward()

        optimizerG.step()

    netG.eval()

    syn_unseen_feature, syn_unseen_label = generate_syn_feature(netG, data.unseenclasses, data.attribute,opt.cls_syn_num)  # 1500x2048
    if opt.gzsl:
        train_X = torch.cat((data.train_feature, syn_unseen_feature), 0)
        train_Y = torch.cat((data.train_label, syn_unseen_label), 0)
        nclass = opt.nclass_all
        v2s = soft_cls.Visual_to_semantic(opt, netS(Variable(train_X.cuda(), volatile=True)).data.cpu(), train_Y,data, nclass, generalized=True)
        opt.gzsl_unseen_output = v2s.unseen_out
        opt.gzsl_seen_output = v2s.seen_out
        cls = classifier2.CLASSIFIER(opt, train_X, train_Y, data, nclass, _beta1=0.5, _nepoch=25, generalized=True)
        print('GZSL Classifier Seen Acc: {:.2f}%, Unseen Acc: {:.2f}%, H Acc: {:.2f}%'.format(cls.seen_cls * 100,cls.unseen_cls * 100,cls.H_cls * 100))
        print('GZSL Ensemble Seen Acc: {:.2f}%, Unseen Acc: {:.2f}%, H Acc: {:.2f}%'.format(cls.seen_ensemble * 100,cls.unseen_ensemble * 100,cls.H_ensemble * 100))
    else:
        fake_syn_unseen_attr = netS(Variable(syn_unseen_feature.cuda(), volatile=True))
        v2s = soft_cls.Visual_to_semantic(opt, fake_syn_unseen_attr.data.cpu(), syn_unseen_label, data,data.unseenclasses.size(0), generalized=False)
        opt.zsl_unseen_output = v2s.output
        cls = classifier2.CLASSIFIER(opt, syn_unseen_feature, util.map_label(syn_unseen_label, data.unseenclasses),data, data.unseenclasses.size(0), _beta1=0.5, _nepoch=25, generalized=False)
        print('ZSL Classifier: {:.2f}%'.format(cls.cls_acc*100))
        print('ZSL Ensemble: {:.2f}%'.format(cls.ensemble_acc*100))
    sys.stdout.flush()
    netG.train()

time_elapsed = time.time() - since
print('End run!!!')
print('Time Elapsed: {}'.format(time_elapsed))
print(GetNowTime())
