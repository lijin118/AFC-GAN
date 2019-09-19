import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys

class CLASSIFIER:
    # train_Y is interger
                 # (train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num,generalized=True, v2s_ratio=opt.v2s_ratio, s2v_ratio=opt.s2v_ratio)
    # def __init__(self,v2s,_train_X, _train_Y, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True,v2s_ratio=0.8,s2v_ratio=0.5):
    def __init__(self,opt, _train_X, _train_Y, data_loader, _nclass, _beta1=0.5, _nepoch=20, generalized=True):
        self.train_X =  _train_X
        self.train_Y = _train_Y 
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.batch_size = opt.cls_batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.data_loader = data_loader
        self.cuda = True
        self.model =  LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(opt.cls_batch_size, self.input_dim)
        self.label = torch.LongTensor(opt.cls_batch_size)
        self.opt = opt
        self.lr = opt.classifier_lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.classifier_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        # w1 = normalizeFeature(self.w.t()).t()
        # self.getSaeSemanticAcc(w1)
        # self.getSaeVisualAcc(w2)

        if generalized:
            self.seen_cls,self.unseen_cls,self.H_cls,self.seen_ensemble,self.unseen_ensemble,self.H_ensemble = self.fit()
        else:
            self.ensemble_acc,self.cls_acc = self.fit_zsl()

    def getSaeSemanticAcc(self,w):
        unseen_attr = self.data_loader.unseenattributes
        fake_test_attr = torch.mm(self.test_unseen_feature,w)
        dist = self.pairwise_distances(fake_test_attr, unseen_attr )
        pred_idx = torch.min(dist,1)[1]
        pred = self.unseenclasses[pred_idx]
        acc = sum(pred==self.test_unseen_label)/self.test_unseen_label.size()[0]
        print('SAE Semantic Acc: {:.2f}%'.format(acc*100))

    def getSaeVisualAcc(self,w):
        unseen_attr = self.data_loader.unseenattributes #50x312
        attr_visual = torch.mm(unseen_attr, w.t())
        dist = self.pairwise_distances(self.test_unseen_feature, attr_visual)
        pred_idx = torch.min(dist,1)[1]
        pred = self.unseenclasses[pred_idx]
        acc = sum(pred==self.test_unseen_label)/self.test_unseen_label.size()[0]
        print('SAE Visual Acc: {:.2f}%'.format(acc*100))



    def pairwise_distances(self,x, y=None):
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
        best_ensemble_acc = 0
        best_cls_acc = 0
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
            ensemble_acc,cls_acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            if ensemble_acc > best_ensemble_acc:
                best_ensemble_acc = ensemble_acc
            if cls_acc > best_cls_acc:
                best_cls_acc = cls_acc
        return best_ensemble_acc,best_cls_acc

    def fit(self):
        best_seen_cls = 0
        best_unseen_cls = 0
        best_H_cls = 0
        best_seen_ensemble = 0
        best_unseen_ensemble = 0
        best_H_ensemble = 0
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

            seen_ensemble,seen_cls = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses,self.opt.gzsl_seen_output)
            unseen_ensemble,unseen_cls = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses,self.opt.gzsl_unseen_output)
            if (seen_cls == 0) or (unseen_cls == 0):
                H_cls = 0
            else:
                H_cls = 2 * seen_cls * unseen_cls / (seen_cls + unseen_cls)


            if (seen_ensemble == 0) or (unseen_ensemble == 0):
                H_ensemble = 0
            else:
                H_ensemble = 2 * seen_ensemble * unseen_ensemble / (seen_ensemble + unseen_ensemble)

            if H_ensemble > best_H_ensemble:
                best_H_ensemble = H_ensemble
                best_seen_ensemble = seen_ensemble
                best_unseen_ensemble = unseen_ensemble
            if H_cls > best_H_cls:
                best_H_cls = H_cls
                best_seen_cls = seen_cls
                best_unseen_cls = unseen_cls


        return best_seen_cls,best_unseen_cls,best_H_cls,best_seen_ensemble,best_unseen_ensemble,best_H_ensemble
                     
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
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return self.train_X[start:end], self.train_Y[start:end]


    def val_gzsl(self, test_X, test_label, target_classes,v2sout):
        output = self.model(Variable(test_X.cuda(), volatile=True))
        _, classifier_label = torch.max(output.data, 1)
        classifier_acc = self.compute_per_class_acc_gzsl(test_label, classifier_label.cpu(), target_classes)

        ensemble_output = output + self.opt.ensemble_ratio * Variable(v2sout)
        _, predicted_label = torch.max(ensemble_output.data, 1)
        ensemble_acc = self.compute_per_class_acc_gzsl(test_label, predicted_label.cpu(), target_classes)
        return ensemble_acc, classifier_acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class 

    def val(self, test_X, test_label, target_classes):
        output = self.model(Variable(test_X.cuda(), volatile=True))
        _, classifier_label = torch.max(output.data, 1)
        classifier_acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), classifier_label.cpu(), target_classes.size(0))

        ensemble_output = output + self.opt.ensemble_ratio * Variable(self.opt.zsl_unseen_output)
        _, predicted_label = torch.max(ensemble_output.data, 1)
        ensemble_acc = self.compute_per_class_acc(util.map_label(test_label, target_classes), predicted_label.cpu(),target_classes.size(0))
        return ensemble_acc,classifier_acc



    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
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
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x):
        o = self.logic(self.fc(x))
        return o  
