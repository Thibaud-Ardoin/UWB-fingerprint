# TODO file to remember the important tasks to go, the ideas and the current focus

- Model Save and load 
- Have a way to force the saving of the current train (?)
- properly understand and setup a sweeping frame
- Dataset visualisation / Understanding tool (?)

- Propose new architectures:
    - Adversarial position learning
    - Reset CNN model
        - Use MaxPool dim reduction
    - 16x16 signals is worth a 1000 words
    - Sliding window

- Change used optimizer:
    - Check Adamdffffff
    - Check SGD the OG of optimizers
    - Warmup ?

- Training process:
    - Try triplet loss again (eventually with covariance regularisation)
    - Could compile covariance on each position or each device individually

- Normalise loss according to batch sizes and embedding size
    - Make sure that this really works (looks like or now it doesnt..)

- Separate better the train and test and validation 
    - just split also test 1st
    - Create a subset for validation (same and different training positions)

- Find other nice metrics to use for evaluating model
    - Create validation process with unseen device
    - NN from on unseen position to an other ( needs multi pose evalutaiton alreaddy)

- Log/ plot architecture of the model for each run

- Embedding size in train is not right

- Terminate runs if they are too bad:
    - Make early stopping conditions
    - could be an early test if the loss is not mooving a lot and then decide according to the test result ?

- Keep the best performing random seeds

- Cross validate on positions while training

- Weight decay

- get compiled parameters such as (total number of trainable parameters, total lambda value)

# Understandings

- Understand why a smaller extender seems to work better ?
- Is the covariance part of the loss incorrect ?