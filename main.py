from deepsleepnet_DLH.pretrain_finetune.deepsleepnet_main import run_pretrain_finetune_deepsleepnet
from deepsleepnet_DLH.sleepKD.sleepKD_main import run_sleepkd_deepsleepnet

import os
import logging
import time

# disable TF warnings in consolep
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_Level"] = "3"


def main():
    total_start_time = time.time()
    run_pretrain_finetune_deepsleepnet()
    # run_sleepkd_deepsleepnet()
    total_duration = total_start_time - time.time()
    print("Took {:.3f}s to run all of run_pretrain_finetune_deepsleepnet)".format(
        total_duration))


main()
