# Basic fast attention mechanisms


To run, run the prepare script first:
`chmod +x prepare.sh && ./prepare.sh`

This will build a virtual environment and install all the dependencies required to run.

Next, to train a transformer run the run script:
`chmod +x run.sh && nohup ./run.sh &`

It's important to asynchrnously run the script via nohup to ensure that training completes.