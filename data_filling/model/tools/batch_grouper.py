from collections import defaultdict

from collections import defaultdict

def group_tags_by_batch(tag_config: dict):
    """
    Regroupe les colonnes selon leur frame_method + frames_used + split_possible + audio.
    Retourne : list of (batch_key, batch_config)
    """
    batches = defaultdict(dict)
    for tag_name, conf in tag_config.items():
        batch_key = (
            conf.get("frame_method"),
            conf.get("frames_used"),
            conf.get("split_possible"),
            conf.get("audio", None)  # on ajoute l'audio même si None
        )
        batches[batch_key][conf["key"]] = conf
    return list(batches.items())
