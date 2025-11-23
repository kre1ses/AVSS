import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    if dataset_items[0]["s1_spectrogram"] is not None:
        s1_specs = torch.stack([item["s1_spectrogram"] for item in dataset_items])
        s1_audios = torch.stack([item["s1_audio"].squeeze(0) for item in dataset_items])
    else:
        s1_specs = None
        s1_audios = None
    if dataset_items[0]["s2_spectrogram"] is not None:
        s2_specs = torch.stack([item["s2_spectrogram"] for item in dataset_items])
        s2_audios = torch.stack([item["s2_audio"].squeeze(0) for item in dataset_items])
    else:
        s2_specs = None
        s2_audios = None

    mix_specs = torch.stack([item["mix_spectrogram"] for item in dataset_items])

    mix_audios = torch.stack([item["mix_audio"].squeeze(0) for item in dataset_items])

    s1_embs = torch.stack(
        [
            torch.from_numpy(item["s1_emb"])
            if item["s1_emb"] is not None
            else torch.tensor(0.0)
            for item in dataset_items
        ]
    )

    s2_embs = torch.stack(
        [
            torch.from_numpy(item["s2_emb"])
            if item["s2_emb"] is not None
            else torch.tensor(0.0)
            for item in dataset_items
        ]
    )

    audio_paths = [item["mix_path"] for item in dataset_items]
    mouth1_paths = [item["mouth1_path"] for item in dataset_items]
    mouth2_paths = [item["mouth2_path"] for item in dataset_items]
    return {
        "s1_spectrogram": s1_specs.squeeze(1) if s1_specs is not None else None,
        "s2_spectrogram": s2_specs.squeeze(1) if s1_specs is not None else None,
        "mix_spectrogram": mix_specs.squeeze(1),
        "s1_audio": s1_audios,
        "s2_audio": s2_audios,
        "mix_audio": mix_audios,
        "s1_embs": s1_embs,
        "s2_embs": s2_embs,
        "mix_path": audio_paths,
        "mouth1_path": mouth1_paths,
        "mouth2_path": mouth2_paths,
    }
