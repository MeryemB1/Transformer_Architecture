from pathlib import Path

def get_config():
    """
    Renvoie la configuration du modèle.
    """
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 256,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    """
    Renvoie le chemin du fichier de poids du modèle pour une epoch donnée.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    """
    Renvoie le chemin du dernier fichier de poids du modèle dans le dossier des poids.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
