"""Optuna search-then-retrain helpers shared by the BreastMNIST, SynapseMNIST3D and VesselMNIST3D notebooks.

The module imports only optuna. The notebook passes in callables that build the model, the data loaders, and the loss
and optimizer, so nothing here depends on torch models, libauc or medmnist.

Typical usage from a notebook (a sketch of the notebooks' code; make_augment_conv3d is defined in the Synapse and
Vessel notebooks):

    def build_model(trial):
        model = ResNet20(pretrained=False, last_activation=None, num_classes=1)
        model.conv1 = torch.nn.Conv2d(X_train.shape[1], 16, kernel_size=3, stride=1, padding=1, bias=False)
        return model

    def build_loaders(trial):
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        train_d = OnTheFlyAugment(train_dataset1, make_augment_conv3d([flip_aug, noise_aug]))
        sampler = DualSampler(train_d, batch_size=batch_size, labels=train_dataset1.labels, sampling_rate=0.5)
        trainloader = data.DataLoader(dataset=train_d, batch_size=batch_size, sampler=sampler)
        trainloader_eval = data.DataLoader(train_d, batch_size=batch_size, shuffle=False, num_workers=2)
        valloader = data.DataLoader(dataset=val_d, batch_size=batch_size, shuffle=False, num_workers=2)
        return trainloader, trainloader_eval, valloader

    def build_loss_optimizer(trial, model):
        lr = trial.suggest_float('lr', 2e-2, 1.5e-1, log=True)
        margin = trial.suggest_float('margin', 0.8, 1.3)
        loss_fn = AUCMLoss_V2(margin=margin)
        optimizer = PESG(model, loss_fn=loss_fn, lr=lr, momentum=0.9, margin=margin, ...)
        return loss_fn, optimizer

    def score_fn(model, valloader, train_fn_result):
        train_log, val_log = train_fn_result
        return max(val_log)

    study = run_optuna_search(build_model, build_loaders, build_loss_optimizer, train,
                               score_fn, n_trials=30, search_epochs=10,
                               search_decay_epochs=[5, 8], search_patience=4)

    model, loss_fn, optimizer, result = retrain_with_best_params(
        study, build_model, build_loaders, build_loss_optimizer, train,
        full_epochs=50, full_decay_epochs=[25, 40], full_patience=12)
"""

import optuna


def run_optuna_search(build_model, build_loaders, build_loss_optimizer, train_fn, score_fn,
                       n_trials, search_epochs, search_decay_epochs, search_patience,
                       direction='maximize', n_startup_trials=10, seed=None,
                       catch=(ValueError,)):
    """Runs an Optuna search and returns the finished Study.

    build_model(trial) -> a new model. Nothing is reused across trials.
    build_loaders(trial) -> (trainloader, trainloader_eval, valloader), built again for every trial. Batch size, sampler
        and so on may use trial.suggest_* calls.
    build_loss_optimizer(trial, model) -> (loss_fn, optimizer) for this trial's model.
    train_fn: dataCentric_functions.train or multilabel_train. Both take decay_epochs. train_with_logits_loss does not, so it
        can't be used here. Called as train_fn(model, loss_fn, optimizer, search_epochs, trainloader, trainloader_eval,
        valloader, patience=search_patience, decay_epochs=search_decay_epochs).
    score_fn(model, valloader, train_fn_result) -> float, the value the study optimizes. The tuning metric is chosen here,
        e.g. max(val_log), or dataCentric_functions.evaluate_val_auc(model, valloader, metric='pr_auc').
        train_fn has already loaded its best-validation-ROC-AUC weights into the model when score_fn runs.
        train_fn_result is what train_fn returned: (train_log, val_log) for train, None for multilabel_train.
    direction: 'maximize' (default) or 'minimize'.
    n_startup_trials: random trials before TPE starts guiding the search (default 10). Keep n_trials well above it.
    seed: TPESampler seed. None (default) leaves the search unseeded.
    catch: exceptions that fail one trial without stopping the study. The default (ValueError,) covers a run that diverges
        to NaN, since roc_auc_score then raises ValueError (single output column; with several columns libauc's
        auc_roc_score skips a failing column). It also hides any other ValueError raised inside a build_* function.
    """
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)

    def objective(trial):
        model = build_model(trial)
        trainloader, trainloader_eval, valloader = build_loaders(trial)
        loss_fn, optimizer = build_loss_optimizer(trial, model)
        result = train_fn(model, loss_fn, optimizer, search_epochs, trainloader, trainloader_eval,
                           valloader, patience=search_patience, decay_epochs=search_decay_epochs)
        return score_fn(model, valloader, result)

    study.optimize(objective, n_trials=n_trials, catch=catch)
    return study


def retrain_with_best_params(study, build_model, build_loaders, build_loss_optimizer, train_fn,
                              full_epochs, full_decay_epochs, full_patience):
    """Trains one final model with study.best_params for the full epoch budget.

    Everything is built through optuna.trial.FixedTrial(study.best_params), so the search and the retrain share the same
    build_* callables. Every parameter they suggest must be in best_params. train_fn must accept decay_epochs, as in the search.

    Returns (model, loss_fn, optimizer, train_fn_result).
    """
    trial = optuna.trial.FixedTrial(study.best_params)
    model = build_model(trial)
    trainloader, trainloader_eval, valloader = build_loaders(trial)
    loss_fn, optimizer = build_loss_optimizer(trial, model)
    result = train_fn(model, loss_fn, optimizer, full_epochs, trainloader, trainloader_eval,
                       valloader, patience=full_patience, decay_epochs=full_decay_epochs)
    return model, loss_fn, optimizer, result
