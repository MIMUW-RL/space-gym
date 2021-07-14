import hashlib
import json


def dict_hash(dictionary: dict) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def make_experiment_hash(model_hyperparams, env_params=None):
    env_params = env_params or dict()
    model_hyperparams = model_hyperparams.copy()
    if "logger_kwargs" in model_hyperparams:
        del model_hyperparams["logger_kwargs"]
    del model_hyperparams["seed"]
    del model_hyperparams["save_freq"]
    model_ac_kwargs = model_hyperparams["ac_kwargs"]
    del model_hyperparams["ac_kwargs"]
    model_hyperparams.update({f"AC_KWARGS_{k}": v for k, v in model_ac_kwargs.items()})
    env_params = {f"ENV_{k}": v for k, v in env_params.items()}
    experiment_dict = dict(**model_hyperparams, **env_params)
    return dict_hash(experiment_dict)
