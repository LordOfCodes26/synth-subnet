module.exports = {
  apps: [
    {
      name: 'miner_1',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 50 --logging.debug --wallet.name wallet --wallet.hotkey synth1 --axon.port 8091 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000 --blacklist.validator_exceptions 0 1 8 17 34 49 53 114 131',
      env: {
        PYTHONPATH: '.'
      },
    },
    {
      name: 'miner_2',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 50 --logging.debug --wallet.name wallet --wallet.hotkey synth2 --axon.port 8092 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000 --blacklist.validator_exceptions 0 1 8 17 34 49 53 114 131',
      env: {
        PYTHONPATH: '.'
      },
    },
    {
      name: 'miner_3',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 50 --logging.debug --wallet.name wallet --wallet.hotkey synth3 --axon.port 8093 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000 --blacklist.validator_exceptions 0 1 8 17 34 49 53 114 131',
      env: {
        PYTHONPATH: '.'
      },
    },
  ],
};
