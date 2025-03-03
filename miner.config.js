module.exports = {
  apps: [
    {
      name: 'miner',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 247 --logging.debug --subtensor.network test --wallet.name miner --wallet.hotkey test4 --axon.port 8091 --blacklist.force_validator_permit true --blacklist.validator_min_stake 1000 --blacklist.validator_exceptions 0 1 8 17 34 49 53 114 131',
      env: {
        PYTHONPATH: '.'
      },
    },
  ],
};
