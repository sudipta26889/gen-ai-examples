import torch
from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=1):
    """Create dataloader with all dataset keys."""
    def collate_fn(batch):
        return {
            'text_ids': torch.stack([item['text_ids'] for item in batch]),
            'text_masks': torch.stack([item['text_mask'] for item in batch]),
            'captions': [item['caption'] for item in batch],
            'video_frames': torch.stack([item['video_frames'] for item in batch]),
            'video_ids': [item['video_id'] for item in batch],
            'videos': [item['video'] for item in batch],
            'sources': [item['source'] for item in batch],
            'categories': [item['category'] for item in batch],
            'urls': [item['url'] for item in batch],
            'start_times': [item['start_time'] for item in batch],
            'end_times': [item['end_time'] for item in batch],
            'ids': [item['id'] for item in batch]
        }
    
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
