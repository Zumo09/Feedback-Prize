from pathlib import Path
from types import SimpleNamespace
from main import main

def launch():
    args = SimpleNamespace(
        lr=1e-4,
        batch_size=16,
        weight_decay=0,
        start_epoch=0,
        epochs=2,
        lr_drop=1,
        clip_max_norm=0.1,
        train_trans_from_epoch=0,
        transformer_lr=1e-5,
        
        # Model parameters
        hidden_dim=4096,
        num_queries=40,
        class_depth=3,
        bbox_depth=5,
        frozen_weights=None,
        resume=None,
        init_last_biases=True,
        dropout=0,
        init_weight=None,
        pretrained=True,

        # Loss coefficients
        losses=["labels", "boxes", "cardinality"],
        ce_loss_coef=1,
        block_ce_for=-1,
        bbox_loss_coef=1, 
        giou_loss_coef=0.5,
        overlap_loss_coef=0.5,
        focal_loss_gamma=2,
        no_class_weight=False,
        effective_num=False,
        beta=0, 

        # Dataset parameters
        input_path="./input/feedback-prize-2021/",
        test_size=0.2,
        no_preprocessing=False,
        num_workers=2,
        dataset_size=1.0, 
        no_align_target=False,

        # Other parameters
        output_dir="./outputs", 
        device="cuda",
        seed=42,
        eval=False,
    )

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


if __name__ == '__main__':
    launch()