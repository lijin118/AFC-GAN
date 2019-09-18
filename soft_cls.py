import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler
import sys

# train_X = torch.cat((data.train_feature, syn_unseen_feature), 0)
# train_Y = torch.cat((data.train_label, syn_unseen_label), 0)
# nclass = opt.nclass_all
# v2s = saev2s.Visual_to_semantic(w2.data.cpu(), train_X, train_Y, data, nclass, generalized=True)

# Visual_to_semantic(opt,train_X, train_Y, data, nclass, generalized=True)
# Visual_to_semantic(opt,syn_unseen_feature, syn_unseen_label, data, data.unseenclasses.size(0), generalized=False)
class Visual_to_semantic:
    # train_Y is interger
    def __init__(self, opt, _train_X, _train_Y, data_loader, _nclass, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100, generalized=False):
        self.train_X = _train_X # 15000x2048
        if generalized:
            self.train_Y = _train_Y
        else:
            self.train_Y = util.map_label(_train_Y, data_loader.unseenclasses)  # 15000
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass #200 or 50
        self.input_dim = data_loader.attribute.size(1)
        self.data_loader = data_loader
        self.attr_dim = self.data_loader.attribute.size(1) #312
        self.cuda = True
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)#in:312, out:200
        self.model.apply(util.weights_init)

        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) #100x312
        self.label = torch.LongTensor(_batch_size) #312
        self.opt = opt

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_Y.size()[0] #19557

        # if generalized:
        #     self.getGZSLAcc()
        # else:
        #     self.getZSLAcc(self.opt.fake_test_attr,self.data_loader.unseenattributes,'Forward KNN Acc')

        if generalized:
            self.acc_seen,self.seen_out, self.acc_unseen,self.unseen_out, self.H = self.fit()
            # print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
            print('V2S Softmax Seen Acc:%.2f, Unseen Acc:%.2f, H Acc:%.2f' % (self.acc_seen * 100,self.acc_unseen * 100,self.H * 100))
        else:
            self.acc,self.output = self.fit_zsl()
            print('V2S Softmax   : %.2f' % (self.acc*100))

    def getZSLAcc(self, fake_test_attr,unseen_attr,outword):
        dist = self.pairwise_distances(fake_test_attr, unseen_attr)
        pred_idx = torch.min(dist, 1)[1]
        pred = self.unseenclasses[pred_idx]
        acc = sum(pred == self.data_loader.test_unseen_label) / self.data_loader.test_unseen_label.size()[0]
        print((outword+': {:.2f}').format(acc * 100))

    def getGZSLAcc(self):
        # fake_test_unseen_attr = torch.mm(self.data_loader.test_unseen_feature, w)
        dist = self.pairwise_distances(self.opt.fake_test_unseen_attr, self.data_loader.attribute)
        pred_idx = torch.min(dist, 1)[1]
        # pred = self.data_loader.unseenclasses[pred_idx]
        acc1 = sum(pred_idx == self.data_loader.test_unseen_label) / self.data_loader.test_unseen_label.size()[0]

        # fake_test_seen_attr = torch.mm(self.data_loader.test_seen_feature, w)
        dist = self.pairwise_distances(self.opt.fake_test_seen_attr, self.data_loader.attribute)
        pred_idx = torch.min(dist, 1)[1]
        # pred = self.data_loader.seenclasses[pred_idx]
        acc2 = sum(pred_idx == self.data_loader.test_seen_label) / self.data_loader.test_seen_label.size()[0]
        if (acc1 == 0) or (acc2 == 0):
            H = 0
        else:
            H = 2 * acc1 * acc2 / (acc1 + acc2)
        print('Forward KNN Seen:{:.2f}%, Unseen:{:.2f}%, H:{:.2f}%'.format(acc2 * 100,acc1 * 100,H * 100))

    def getSaeVisualAcc(self, w):
        unseen_attr = self.data_loader.unseenattributes  # 50x312
        attr_visual = torch.mm(unseen_attr, w.t())
        dist = self.pairwise_distances(self.test_unseen_feature, attr_visual)
        pred_idx = torch.min(dist, 1)[1]
        pred = self.unseenclasses[pred_idx]
        acc = sum(pred == self.test_unseen_label) / self.test_unseen_label.size()[0]
        print('SAE Visual Acc: {:.2f}%'.format(acc * 100))

    def pairwise_distances(self, x, y=None):
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

    def fit_zsl(self):
        best_acc = 0
        best_output = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
                # print('Training classifier loss= ', loss.data[0])
            acc,output = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            # print('acc %.4f' % (acc))
            if acc >= best_acc:
                best_acc = acc
                best_output = output
        return best_acc,best_output

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_seen_out = None
        best_unseen_out = None
        for epoch in range(self.nepoch):
            self.model.train()
            for i in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input) #100x312
                labelv = Variable(self.label)
                output = self.model(inputv) #100x50
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
                # print('Training classifier loss= ', loss.data[0])
            self.model.eval()
            acc_seen,seen_out = self.val_gzsl(self.opt.fake_test_seen_attr, self.test_seen_label, self.seenclasses)
            acc_unseen,unseen_out = self.val_gzsl(self.opt.fake_test_unseen_attr, self.test_unseen_label, self.unseenclasses)
            if (acc_seen == 0) or (acc_unseen == 0):
                H = 0
            else:
                H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
            # print('acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (acc_seen, acc_unseen, H))
            if H >= best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                best_seen_out = seen_out
                best_unseen_out = unseen_out
        return best_seen,best_seen_out, best_unseen, best_unseen_out, best_H

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            # print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    def val_gzsl(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_output = torch.FloatTensor(test_label.size(0), self.nclass).cuda()
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                output = self.model(Variable(test_X[start:end].cuda(), volatile=True))
            else:
                output = self.model(Variable(test_X[start:end], volatile=True))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            all_output[start:end, :] = output.data
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc,all_output

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class

        # test_label is integer

    def val(self, test_X, test_label, target_classes):
        fake_test_attr = self.opt.fake_test_attr
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        all_output = torch.FloatTensor(test_label.size(0),self.nclass).cuda()
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            if self.cuda:
                output = self.model(Variable(fake_test_attr[start:end].cuda(), volatile=True))
            else:
                output = self.model(Variable(fake_test_attr[start:end], volatile=True))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            all_output[start:end,:] = output.data
            start = end

        acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label,
                                         target_classes.size(0))
        return acc,all_output

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
        return acc_per_class.mean()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        # self.relu = nn.ReLU(True)
        self.logic = nn.LogSoftmax(dim=1)
        # self.apply(weights_init)

    def forward(self, x):
        # o = self.relu(self.fc(x))
        # o = self.logic(o)

        o = self.logic(self.fc(x))
        return o
