"""Download and preprocess all datasets used in paper experiments."""

from data.process import ecg_processdata
from data.process import har_processdata
from data.process import ppg_processdata
from data.process import sleepeeg_processdata

# Linear probe / semi-supervised pretraining datasets
print("Downloading and processing pretraining datasets...")
har_processdata.main()
ecg_processdata.main()
ppg_processdata.main()
sleepeeg_processdata.main()
