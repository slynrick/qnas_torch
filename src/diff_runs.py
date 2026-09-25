""" Compare the effective configuration of two runs (or of a config and a run).

Each side is one of:
  - an experiment directory: its log_params_evolution.txt is used;
  - a log_params_*.txt file;
  - a .yml config: rendered through run_evolution's CLI and ConfigParameters exactly as
    an evolve run would record it. Some QNAS settings come from CLI flags, not the YAML
    (run_pipeline.sh passes -X as --en_pop_crossover, -m as --fitness_metric), so pass
    the evolve flags the run will use with --evolve-args.

Paths and provenance (the files section, experiment/data paths, phase, log level) are
ignored. Exit status is 0 when the configurations match and 1 otherwise, like diff.

    uv run python src/diff_runs.py experiment_cifar10_progressive/exp4 experiment_cifar10_progressive/exp5
    uv run python src/diff_runs.py configs/config_files_cifar/config_progressive.yml \\
        experiment_cifar10_progressive/exp4 --evolve-args "--fitness_metric best_accuracy --en_pop_crossover"
"""

import argparse
import io
import os
import shlex
import sys

import yaml

IGNORED_SECTIONS = ('files',)
IGNORED_KEYS = {'train.experiment_path', 'train.data_path', 'train.phase', 'train.log_level'}


def render_config(config_file, evolve_args=''):
    """ Parameters an evolve run of *config_file* with *evolve_args* would write to its
        log_params_evolution.txt, parsed back the same way (through the same printer). """
    # Imported lazily: rendering validates the op menu against cnn.model (torch), which
    # comparing two existing runs does not need.
    import qnas_config as cfg
    import run_evolution

    argv = ['--experiment_path', '<render>', '--data_path', '<render>', '--dataset', 'cifar10',
            '--network_config', 'default', '--config_file', config_file]
    args = vars(run_evolution.build_parser().parse_args(argv + shlex.split(evolve_args)))
    config = cfg.ConfigParameters(args, phase='evolution')
    config.get_parameters()
    _, params_dict = config.logfile_params()
    buffer = io.StringIO()
    config.params_to_logfile(params_dict, buffer)
    return yaml.safe_load(buffer.getvalue())


def load_params(source, evolve_args=''):
    if os.path.isdir(source):
        source = os.path.join(source, 'log_params_evolution.txt')
    if source.endswith(('.yml', '.yaml')):
        return render_config(source, evolve_args)
    with open(source) as f:
        return yaml.safe_load(f)


def flatten(tree, prefix=''):
    """ {'QNAS': {'a': 1}} -> {'QNAS.a': 1}; non-empty nested dicts are expanded. """
    flat = {}
    for key, value in tree.items():
        path = f'{prefix}{key}'
        if isinstance(value, dict) and value:
            flat.update(flatten(value, path + '.'))
        else:
            flat[path] = value
    return flat


def relevant(flat):
    return {key: value for key, value in flat.items()
            if key.split('.')[0] not in IGNORED_SECTIONS and key not in IGNORED_KEYS}


def diff_params(a, b):
    """ Return (changed, only_a, only_b) between two parsed log_params trees. """
    fa, fb = relevant(flatten(a)), relevant(flatten(b))
    changed = {key: (fa[key], fb[key]) for key in fa.keys() & fb.keys() if fa[key] != fb[key]}
    only_a = {key: fa[key] for key in fa.keys() - fb.keys()}
    only_b = {key: fb[key] for key in fb.keys() - fa.keys()}
    return changed, only_a, only_b


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Diff the effective configuration of two runs or configs.')
    parser.add_argument('a', help='experiment dir, log_params_*.txt or .yml config')
    parser.add_argument('b', help='experiment dir, log_params_*.txt or .yml config')
    parser.add_argument('--evolve-args', default='',
                        help='run_evolution.py flags used to render a .yml side, e.g. '
                             '"--en_pop_crossover --fitness_metric best_accuracy"')
    args = parser.parse_args(argv)

    changed, only_a, only_b = diff_params(load_params(args.a, args.evolve_args),
                                          load_params(args.b, args.evolve_args))
    if not (changed or only_a or only_b):
        print('Same configuration.')
        return 0

    if changed:
        print('changed (A -> B):')
        for key in sorted(changed):
            print(f'  {key}: {changed[key][0]} -> {changed[key][1]}')
    for label, only in (('only in A:', only_a), ('only in B:', only_b)):
        if only:
            print(label)
            for key in sorted(only):
                print(f'  {key}: {only[key]}')
    if only_a or only_b:
        print('(a key on one side only often means that run predates the option and ran '
              'with its default - check the default before treating it as a difference)')
    return 1


if __name__ == '__main__':
    sys.exit(main())
