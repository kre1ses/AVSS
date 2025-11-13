import torch
from torch.nn.utils.rnn import pad_sequence


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

    s1_specs = [item["s1_spectrogram"] for item in dataset_items]
    s2_specs = [item["s2_spectrogram"] for item in dataset_items]
    mix_specs = [item["mix_spectrogram"] for item in dataset_items]

    s1_spec_lengths = torch.tensor([s.shape[0] for s in s1_specs], dtype=torch.long)
    s2_spec_lengths = torch.tensor([s.shape[0] for s in s2_specs], dtype=torch.long)
    mix_spec_lengths = torch.tensor([s.shape[0] for s in mix_specs], dtype=torch.long)

    s1_spec_batch = pad_sequence(s1_specs, batch_first=True, padding_value=0.0)
    s2_spec_batch = pad_sequence(s2_specs, batch_first=True, padding_value=0.0)
    mix_spec_batch = pad_sequence(mix_specs, batch_first=True, padding_value=0.0)

    s1_audios = pad_sequence([item["s1_audio"].squeeze(0) for item in dataset_items])
    s2_audios = pad_sequence([item["s2_audio"].squeeze(0) for item in dataset_items])
    mix_audios = pad_sequence([item["mix_audio"].squeeze(0) for item in dataset_items])

    # s1_mouths = torch.tensor([torch.tensor(item["mouth1"]) for item in dataset_items])
    s1_mouths = torch.stack(
        [torch.from_numpy(item["mouth1"]) for item in dataset_items]
    )
    # s2_mouths = torch.tensor([torch.tensor(item["mouth2"]) for item in dataset_items])
    s2_mouths = torch.stack(
        [torch.from_numpy(item["mouth2"]) for item in dataset_items]
    )

    audio_paths = [item["mix_path"] for item in dataset_items]
    mouth1_paths = [item["mouth1_path"] for item in dataset_items]
    mouth2_paths = [item["mouth2_path"] for item in dataset_items]

    return {
        "s1_spectrogram": s1_spec_batch,
        "s2_spectrogram": s2_spec_batch,
        "mix_spectrogram": mix_spec_batch,
        "s1_spectrogram_lenghts": s1_spec_lengths,
        "s2_spectrogram_lenghts": s2_spec_lengths,
        "mix_spectrogram_lenghts": mix_spec_lengths,
        "s1_audio": s1_audios,
        "s2_audio": s2_audios,
        "mix_audio": mix_audios,
        "s1_mouth": s1_mouths,
        "s2_mouth": s2_mouths,
        "mix_path": audio_paths,
        "mouth1_path": mouth1_paths,
        "mouth2_path": mouth2_paths,
    }
