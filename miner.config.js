module.exports = {
  apps: [
    {
      name: 'miner_1',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 50 --logging.debug --wallet.name wallet --wallet.hotkey synth1 --axon.port 8091',
      env: {
        PYTHONPATH: '.'
      },
    },
    {
      name: 'miner_2',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 50 --logging.debug --wallet.name wallet --wallet.hotkey synth2 --axon.port 8092',
      env: {
        PYTHONPATH: '.'
      },
    },
    {
      name: 'miner_3',
      interpreter: 'python3',
      script: './neurons/miner.py',
      args: '--netuid 50 --logging.debug --wallet.name wallet --wallet.hotkey synth3 --axon.port 8093',
      env: {
        PYTHONPATH: '.'
      },
    },
  ],
};
