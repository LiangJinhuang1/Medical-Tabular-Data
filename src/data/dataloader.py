from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size, shuffle, config):
    """Create DataLoader with config"""
    loader_cfg = config.get('loader', {})
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=loader_cfg.get('num_workers', 0),
        pin_memory=loader_cfg.get('pin_memory', False),
        prefetch_factor=loader_cfg.get('prefetch_factor', None),
        persistent_workers=loader_cfg.get('persistent_workers', False),
    )