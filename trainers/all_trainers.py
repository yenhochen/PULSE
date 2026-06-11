from trainers.pulse import PULSETrainer
from trainers.simclr import SimCLRTrainer
from trainers.dsvae import DSVAETrainer

from trainers.rebar import RebarTrainer
from trainers.ts2vec import TS2VecTrainer
from trainers.patchtst import PatchTSTTrainer
from trainers.timeMAE import TimeMAETrainer
from trainers.supervised import SupervisedTrainer
from trainers.pulse_oracle import PULSEOracleTrainer

from trainers.lfads import LFADSTrainer
from trainers.transfer import TransferTrainer


all_trainers = {
    # contrastive methods
    "simclr": SimCLRTrainer,
    "ts2vec": TS2VecTrainer,
    "rebar": RebarTrainer,
    "dsvae": DSVAETrainer,
    "lfads": LFADSTrainer,
    "timeMAE": TimeMAETrainer,
    "patchtst": PatchTSTTrainer,
    "pulse": PULSETrainer,
    "pulse_oracle": PULSEOracleTrainer,
    "supervised": SupervisedTrainer,
    "transfer": TransferTrainer,
}

