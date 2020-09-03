import os
import torch


def load_state_simple(path, model, ignore=[]):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        if len(ignore) > 0:
            for k in list(checkpoint.keys()):
                flag = False
                for prefix in ignore:
                    if k.startswith(prefix):
                        flag = True
                        the_prefix = prefix
                        break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del checkpoint[k]
        model.load_state_dict(checkpoint, strict=False)

        keys1 = set(checkpoint.keys())
        keys2 = set([k for k, _ in model.named_parameters()])
        not_loaded = keys2 - keys1
        for k in not_loaded:
            print('caution: {} not loaded'.format(k))
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)


def load_state(path, model, ignore=[], optimizer=None):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        if len(ignore) > 0:
            assert optimizer == None
            for k in list(checkpoint['state_dict'].keys()):
                flag = False
                for prefix in ignore:
                    if k.startswith(prefix):
                        flag = True
                        the_prefix = prefix
                        break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del checkpoint['state_dict'][k]
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        keys1 = set(checkpoint['state_dict'].keys())
        keys2 = set([k for k, _ in model.named_parameters()])
        not_loaded = keys2 - keys1
        for k in not_loaded:
            print('caution: {} not loaded'.format(k))

        if optimizer != None:
            assert len(ignore) == 0
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (step {})".format(path, checkpoint['step']))
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)
