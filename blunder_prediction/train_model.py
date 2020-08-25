import os
import os.path
import argparse
import time

import numpy as np
import humanize
import yaml

import maia_chess_backend
import maia_chess_backend.torch

import torch
import torch.utils.tensorboard

np.random.seed(12345)

@maia_chess_backend.profile_helper
def main():
    parser = argparse.ArgumentParser(description='Train a NN predictor from config', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config', help='config file for model / training')
    args = parser.parse_args()

    tstart = time.time()
    with open(args.config) as f:
        config = yaml.safe_load(f.read())

    collection_name = os.path.basename(os.path.dirname(args.config)).replace('configs_', '')
    name = os.path.basename(args.config).split('.')[0]
    outputDir = os.path.join('models', collection_name, name)
    os.makedirs(outputDir, exist_ok = True)
    tensorboard_writer = maia_chess_backend.torch.TB_wrapper(name, log_dir = os.path.join('runs', collection_name))

    with torch.cuda.device(config['device']):
        maia_chess_backend.printWithDate(f"Loading model:{config['model']}")
        net = maia_chess_backend.torch.NetFromConfigNew(config['model'])

        train_loader, test_loader, val_loader = setupLoaders(config)
        try:
            train_loop(net, config, train_loader, test_loader, val_loader, tensorboard_writer, outputDir)
        except KeyboardInterrupt:
            net.save(os.path.join(outputDir, f"net-final.pt"))

    maia_chess_backend.printWithDate(f"Done everything in {humanize.naturaldelta(time.time() - tstart)}, exiting")

@maia_chess_backend.profile_helper
def setupLoaders(config):
    if config['training'].get('old_loader', None):
        loader_type = maia_chess_backend.torch.MmapIterLoaderMap_old
    else:
        loader_type = maia_chess_backend.torch.MmapIterLoaderMap
    mmap_loader_train = loader_type(
                    config['dataset']['input_train'],
                    config['model']['outputs'] + config['model'].get('inputs', []),
                    config['training']['batch_size'],
                    max_samples = config['training'].get('max_samples', None),
                    )
    mmap_loader_test = loader_type(
                    config['dataset']['input_test'],
                    config['model']['outputs'] + config['model'].get('inputs', []),
                    config['training']['batch_size'],
                    max_samples = config['training'].get('max_samples', None),
                    )
    if 'input_validate' in config['dataset']:
        val_path = config['dataset']['input_validate']
    else:
        val_path = config['dataset']['input_test']
    mmap_loader_val = maia_chess_backend.torch.MmapIterLoaderMap(
                    val_path,
                    ['is_blunder_wr', 'winrate_loss'] + config['model'].get('inputs', []),
                    config['training']['batch_size'],
                    max_samples = config['training'].get('max_samples', None),
                    linear_mode = True,
                    )

    return mmap_loader_train, mmap_loader_test, mmap_loader_val

@maia_chess_backend.profile_helper
def train_loop(net, config, train_loader, test_loader, val_loader, tensorboard_writer, outputDir):
    maia_chess_backend.printWithDate(f"Starting training loop")
    if torch.cuda.is_available():
        net.cuda()

    lastFewAcs = []

    optimizer = torch.optim.Adam(
                    net.parameters(),
                    lr = config['training']['lr_intial'],
                    #momentum = 0.9,
                    weight_decay = 0.0001,
                    betas = (0.9, 0.999),
                    eps = 1e-8,
                    #nesterov = True,
                    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=config['training']['lr_steps'],
                    gamma=config['training']['lr_gamma'],
                    )
    loss_reg = torch.nn.MSELoss(reduction='mean')
    loss_class = torch.nn.CrossEntropyLoss(ignore_index = -1)

    epoch_losses = {'count' : 0}
    step_durations = []
    tstart = time.time()
    tests = 0
    last_save = time.time()

    for step in range(config['training']['total_steps']):
        delta_t = step_train(net, train_loader, optimizer, loss_reg, loss_class, epoch_losses)

        step_durations.append(delta_t)

        if step % 100 == 0:
            i = step % config['training']['test_steps']

            maia_chess_backend.printWithDate(f"Step {step} {i /config['training']['test_steps']*100:02.0f}% {make_info_str(epoch_losses)} {(i + 1) / (time.time() - tstart):03.2f} steps/second", end = '\r')

        if step > 0 and step % config['training']['validate_steps'] == 0:

            val_results = step_validate(net, val_loader, config['training']['test_size'] * 10)
        else:
            val_results = None

        if step > 0 and step % config['training']['test_steps'] == 0:
            tests += 1
            maia_chess_backend.printWithDate(f"Training step {step} losses: {make_info_str(epoch_losses)}" + ' ' * 10)

            test_losses, accuracies = step_test(net, test_loader, loss_reg, loss_class, config['training']['test_size'])

            maia_chess_backend.printWithDate(f"Testing {tests} step {step}  losses: {make_info_str(test_losses)} accuracy: {make_info_str(accuracies)}")

            last_save, batch_acc = save_results(step, tests, last_save, net, tensorboard_writer, epoch_losses, test_losses, accuracies, val_results, optimizer, step_durations, config['training']['batch_size'], train_loader.num_blunders, train_loader.num_nonblunders, outputDir)

            if tests == 1:
                t_x, t_y = next(train_loader)
                if net.has_extras:
                    tensorboard_writer.add_graph(
                            net,
                            input_to_model = (t_x, t_y),
                            )
                else:
                    tensorboard_writer.add_graph(net,input_to_model = t_x)
            if config['training'].get('auto_stop', None) is not None and batch_acc is not None:
                lastFewAcs.append(batch_acc)
                if len(lastFewAcs) - np.argmax(lastFewAcs) > config['training'].get('auto_stop', None) - 1:
                    break
            epoch_losses = {'count' : 0}
            step_durations = []
            tstart = time.time()

        scheduler.step()

    test_losses, accuracies = step_test(net, test_loader, loss_reg, loss_class, config['training']['test_size'])
    val_results = step_validate(net, val_loader, config['training']['test_size'] * 10)
    net.save(os.path.join(outputDir, f"net-final-{step}.pt"))
    last_save, batch_acc = save_results(step, tests, last_save, net, tensorboard_writer, epoch_losses, test_losses, accuracies, val_results, optimizer, step_durations, config['training']['batch_size'], train_loader.num_blunders, train_loader.num_nonblunders, outputDir)

@maia_chess_backend.profile_helper
def step_train(net, train_loader, optimizer, loss_reg, loss_class, epoch_losses):
    tstart = time.time()
    x, y = next(train_loader)
    batch_losses = net.train_batch(x, y, optimizer, loss_reg, loss_class)
    epoch_losses['count'] += 1
    for n, v in batch_losses.items():
        try:
            epoch_losses[n] += v
        except KeyError:
            epoch_losses[n] = v
    return time.time() - tstart

@maia_chess_backend.profile_helper
def step_test(net, test_loader, loss_reg, loss_class, num_tests):

    losses_test = {'count' : 0}
    accuracy_test = {'count' : 0}

    for _ in range(num_tests):
        x, y = next(test_loader)

        batch_losses, batch_accuracy = net.test_batch(x, y, loss_reg, loss_class)
        losses_test['count'] += 1
        for n, v in batch_losses.items():
            try:
                losses_test[n] += v
            except KeyError:
                losses_test[n] = v
        accuracy_test['count'] += 1
        for n, v in batch_accuracy.items():
            try:
                accuracy_test[n] += v
            except KeyError:
                accuracy_test[n] = v
    return losses_test, accuracy_test

@maia_chess_backend.profile_helper
def step_validate(net, val_loader, num_tests):
    results = {'count' : 0}
    for _ in range(num_tests):
        x, y = next(val_loader)
        batch_results = net.validate_batch(x, y)
        results['count'] += 1
        for n, v in batch_results.items():
            try:
                results[n] += v
            except KeyError:
                results[n] = v
    return results

@maia_chess_backend.profile_helper
def save_results(step, tests, last_save, net, tensorboard_writer, train_losses, test_losses, test_accuracies, val_results, optimizer, step_durations, boards_per_step, num_blunders, num_nonblunders, outputDir):
    if time.time() - last_save > 60 * 60 * 3:
        net.save(os.path.join(outputDir, f"net-{step}.pt"))
        last_save = time.time()
    rets_name = os.path.join(outputDir, 'training_dat.csv')
    if not os.path.isfile(rets_name):
        with open(rets_name, 'a') as f:
            f.write("step,test_num,step_duration,")
            f.write(','.join([f"training_loss_{c}" for c in sorted(train_losses.keys()) if c != 'count']))
            f.write(',')
            f.write(','.join([f"testing_loss_{c}" for c in sorted(test_losses.keys()) if c != 'count']))
            f.write(',')
            f.write(','.join([f"testing_accuracies_{c}" for c in sorted(test_accuracies.keys()) if c != 'count']))
            f.write(',')
            f.write(','.join(maia_chess_backend.torch.validation_stats))
            f.write('\n')
    batch_acc = None
    with open(rets_name, 'a') as f:
        row_strs = []
        c = train_losses['count']
        for n in sorted(train_losses.keys()):
            if n == 'count':
                continue
            v = train_losses[n] / c
            tensorboard_writer.add_scalar(f"Training/Loss/{n}", v, step)
            row_strs.append(str(v.item()))

        row_strs.append(str(v))
        c = test_losses['count']
        for n in sorted(test_losses.keys()):
            if n == 'count':
                continue
            v = test_losses[n] / c
            tensorboard_writer.add_scalar(f"Testing/Loss/{n}", v, step)
            row_strs.append(str(v.item()))
        c = test_accuracies['count']
        for n in sorted(test_accuracies.keys()):
            if n == 'count':
                continue
            v = test_accuracies[n] / c
            tensorboard_writer.add_scalar(f"Testing/Accuracy/{n}", v, step)
            if n in ['is_blunder_mean', 'is_blunder_wr']:
                batch_acc = v.item()
            row_strs.append(str(v.item()))

        if val_results is not None:
            for n in maia_chess_backend.torch.validation_stats:
                v = val_results[n] / val_results['count']
                tensorboard_writer.add_scalar(f"Validation/{n}", v, step)
                row_strs.append(str(v.item()))

        tensorboard_writer.add_scalar(f"Xtras/Training/Learning_rate", optimizer.param_groups[0]['lr'], step)
        tensorboard_writer.add_scalar(f"Xtras/Training/Mean_step_duration", np.mean(step_durations), step)
        tensorboard_writer.add_scalar(f"Xtras/Training/Boards", step * boards_per_step, step)

        tensorboard_writer.add_scalar(f"Xtras/Training/Ratio_blunders", step * boards_per_step / num_blunders / 2, step)
        tensorboard_writer.add_scalar(f"Xtras/Training/Ratio_nonblunders", step * boards_per_step / num_nonblunders / 2, step)

        for n, a in net.gen_hist_dicts().items():
            tensorboard_writer.add_histogram(n, a, step)

        tensorboard_writer.flush()

        f.write(f"{step},{tests},{np.mean(step_durations)},{','.join(row_strs)}\n")

        return last_save, batch_acc

def make_info_str(dat_dict):
    infos = []
    c = dat_dict['count']
    for n, l in dat_dict.items():
        if n == 'count':
            continue
        infos.append(f"{n} {l / c:.4f}")
    return ','.join(infos)

if __name__ == '__main__':
    main()
