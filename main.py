from deepsleepnet_DLH.pretrain_finetune.deepsleepnet_main import run_pretrain_finetune_deepsleepnet
from deepsleepnet_DLH.sleepKD.sleepKD_main import run_sleepkd_deepsleepnet

import os
import logging

# disable TF warnings in console
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_Level"] = "3"


def main():
    run_pretrain_finetune_deepsleepnet()
    # run_sleepkd_deepsleepnet()


main()
