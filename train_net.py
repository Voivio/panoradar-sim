from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor

from panoradar import get_panoradar_cfg, register_dataset, register_sim_dataset, get_trainer_class

def main(args):
    cfg = get_panoradar_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    TrainClass = get_trainer_class(cfg)
    default_setup(cfg, args)

    is_sim_needed = any(["sim" in cfg.DATASETS.TRAIN[0], "sim" in cfg.DATASETS.TEST[0]])
    if is_sim_needed:
        register_sim_dataset(cfg)

    is_real_needed = any(["sim" not in cfg.DATASETS.TRAIN[0], "sim" not in cfg.DATASETS.TEST[0]])
    if is_real_needed:
        register_dataset(cfg)

    if args.eval_only:
        predictor = DefaultPredictor(cfg)
        res = TrainClass.test(cfg, predictor.model)
        return res

    trainer = TrainClass(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )