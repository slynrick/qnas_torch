"""cnn.model.NetworkGraph: every op in the shipped op menus must build and run on CPU."""

import copy
import os

import numpy as np
import pytest
import torch
import yaml

from cnn import model
from conftest import ROOT_DIR

NUM_CLASSES = 10
INPUT = (2, 3, 32, 32)

MENU_CONFIGS = [
    'configs/config_files_cifar/01_deterministic_13-8-4.yml',
    'configs/config_files_cifar/07_dynamic_v2.yml',
]


def load_fn_dict(rel_path):
    with open(os.path.join(ROOT_DIR, rel_path)) as f:
        raw = yaml.safe_load(f)['QNAS']['function_dict']
    return {name: {'function': d['function'], 'params': dict(d['params'])}
            for name, d in raw.items()}


def all_ops():
    seen, params = set(), []
    for rel in MENU_CONFIGS:
        for name, definition in load_fn_dict(rel).items():
            key = (name, definition['function'], tuple(sorted(definition['params'].items())))
            if key not in seen:
                seen.add(key)
                params.append(pytest.param(rel, name, id=f'{os.path.basename(rel)}::{name}'))
    return params


def build(net_list, fn_dict, network_config='default', network_gap=False):
    net = model.NetworkGraph(num_classes=NUM_CLASSES, network_config=network_config,
                             network_gap=network_gap)
    # create_functions mutates the dicts it is handed (injects in_channels) - the trainer
    # passes a filtered copy, so do the same here.
    net.create_functions(net_list=list(net_list),
                         fn_dict=copy.deepcopy({k: v for k, v in fn_dict.items() if k in net_list}))
    return net


def run(net):
    net.eval()
    with torch.no_grad():
        return net(torch.randn(*INPUT))


@pytest.mark.parametrize('rel, op', all_ops())
def test_single_op_network_produces_logits(rel, op):
    fn_dict = load_fn_dict(rel)
    logits = run(build([op], fn_dict))
    assert logits.shape == (INPUT[0], NUM_CLASSES)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize('rel', MENU_CONFIGS)
def test_random_deep_networks_build_and_run(rel):
    fn_dict = load_fn_dict(rel)
    names = list(fn_dict)
    rng = np.random.default_rng(0)
    for _ in range(4):
        net_list = list(rng.choice(names, size=6))
        logits = run(build(net_list, fn_dict))
        assert logits.shape == (INPUT[0], NUM_CLASSES), net_list


def test_noop_layers_are_skipped():
    fn_dict = {'conv': {'function': 'ConvBlock', 'params': {'kernel': 3, 'strides': 1, 'filters': 8}},
               'no_op': {'function': 'NoOp', 'params': {}}}
    with_noops = build(['no_op', 'conv', 'no_op'], fn_dict)
    plain = build(['conv'], fn_dict)
    assert len(with_noops.layers) == len(plain.layers) == 1


def test_all_noop_network_still_classifies():
    fn_dict = {'no_op': {'function': 'NoOp', 'params': {}}}
    assert run(build(['no_op', 'no_op'], fn_dict)).shape == (INPUT[0], NUM_CLASSES)


def test_channels_are_chained_between_layers():
    fn_dict = {'a': {'function': 'ConvBlock', 'params': {'kernel': 3, 'strides': 1, 'filters': 16}},
               'b': {'function': 'ConvBlock', 'params': {'kernel': 3, 'strides': 1, 'filters': 24}}}
    net = build(['a', 'b'], fn_dict)
    convs = [m for m in net.model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [(c.in_channels, c.out_channels) for c in convs] == [(3, 16), (16, 24)]


def test_fc_is_lazily_created_on_first_forward():
    fn_dict = {'a': {'function': 'ConvBlock', 'params': {'kernel': 3, 'strides': 2, 'filters': 8}}}
    net = build(['a'], fn_dict)
    assert net.fc is None
    run(net)
    assert net.fc is not None


def test_dense_network_config_runs():
    fn_dict = {'a': {'function': 'ConvBlock', 'params': {'kernel': 3, 'strides': 1, 'filters': 8}},
               'p': {'function': 'MaxPooling', 'params': {'kernel': 2, 'strides': 2}}}
    logits = run(build(['a', 'p', 'a'], fn_dict, network_config='dense'))
    assert logits.shape == (INPUT[0], NUM_CLASSES)


def test_invalid_network_config_rejected():
    net = model.NetworkGraph(num_classes=NUM_CLASSES, network_config='bogus')
    with pytest.raises(ValueError, match='Invalid network configuration'):
        net.create_functions(net_list=[], fn_dict={})
